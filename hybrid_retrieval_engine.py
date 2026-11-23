import json
import pandas as pd
import networkx as nx
import numpy as np
from networkx.algorithms import community
from networkx.algorithms.approximation import steiner_tree
import matplotlib
matplotlib.use("Agg")  # headless environment; avoid Tk/Agg backend issues
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter

# Configure a CJK font so subgraph PNG titles/legends render correctly
for _cjk_font in ["Noto Sans CJK JP", "Noto Sans CJK SC", "AR PL UMing CN", "WenQuanYi Micro Hei"]:
    try:
        font_manager.findfont(_cjk_font, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _cjk_font
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


class MultimorbidityRetrievalEngine:
    # ===================== Fast Lane switch =====================
    # Steady-state comorbidity module matching (Fast Lane) is disabled because:
    #   1. The score formula weight_sum / len(union) can never reach the 0.6 threshold -
    #      All 2162 stacking risk values fall in [8.2e-05, 0.0567], score ceiling ~0.057;
    #      Even with all weights set to 1.0, the best score across 235 patients is only 0.522, still unreachable;
    #   2. The module library has only 2 large communities (9 / 20 diseases); a patient vector is usually a subset,
    #      Weighted Jaccard can hardly exceed 0.6;
    #   3. The retrieval trigger is now the risk tier, so Fast Lane's numeric-weight assumption no longer applies.
    # Set this attribute back to True to restore; _match_anchor_library stays unchanged.
    ENABLE_FAST_LANE = False

    def __init__(self, kg_path, s2_path, mapping_path):
        self.kg_path = kg_path
        self.s2_path = s2_path
        self.mapping_path = mapping_path

        self.kg = nx.Graph()
        self.mapping_dict = {}
        self.module_library = []

        self._load_and_build_base_graph()

    def _load_and_build_base_graph(self):
        """Load the full heterogeneous graph and parse the relation_type hierarchy correctly"""
        print("Initializing full heterogeneous graph...")

        # Added: save raw edges data for later full edge-information retrieval
        with open(self.kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        self._original_edges = kg_data.get('edges', {})

        df_map = pd.read_csv(self.mapping_path)
        self.mapping_dict = dict(zip(df_map['Original Entity'], df_map['Standard Disease Name']))
        for sd in set(df_map['Standard Disease Name']):
            self.mapping_dict[sd] = sd

        with open(self.kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)

        relation_type_counter = Counter()
        edge_count = 0
        parse_error_count = 0

        edges_data = kg_data.get('edges', {})

        if isinstance(edges_data, dict):
            edge_items = edges_data.items()
        elif isinstance(edges_data, list):
            edge_items = enumerate(edges_data)
        else:
            raise ValueError("invalid edges field; expected dict or list")

        for edge_id, edge in edge_items:
            src = self.mapping_dict.get(edge.get('source_term', ''), edge.get('source_term', ''))
            tgt = self.mapping_dict.get(edge.get('target_term', ''), edge.get('target_term', ''))

            if not src or not tgt:
                continue

            weight = float(edge.get('weight', 0.5))

            document_sources = edge.get('document_sources', {})
            edge_type = 'ASSOCIATION'

            if isinstance(document_sources, dict) and len(document_sources) > 0:
                relation_types = []
                for doc_id, doc_data in document_sources.items():
                    try:
                        if isinstance(doc_data, dict):
                            rt = doc_data.get('relation_type')
                            if rt:
                                relation_types.append(rt)
                            else:
                                relation_detail = doc_data.get('relation_detail', {})
                                if isinstance(relation_detail, dict):
                                    rt = relation_detail.get('relation_type')
                                    if rt:
                                        relation_types.append(rt)
                    except Exception as e:
                        parse_error_count += 1
                        continue

                if relation_types:
                    edge_type = Counter(relation_types).most_common(1)[0][0]
                    relation_type_counter[edge_type] += 1
                else:
                    relation_type_counter['ASSOCIATION'] += 1
            else:
                edge_type = edge.get('type', 'ASSOCIATION')
                relation_type_counter[edge_type] += 1

            nav_weight = weight
            edge_type_upper = edge_type.upper()
            if any(k in edge_type_upper for k in ['RISK', 'PREDICT', 'CAUSE', 'CAUSAL', 'RISK_FACTOR']):
                nav_weight *= 1.5
            elif any(k in edge_type_upper for k in ['TREAT', 'THERAPY', 'PREVENT']):
                nav_weight *= 1.2
            else:
                nav_weight *= 1.0

            if self.kg.has_edge(src, tgt):
                self.kg[src][tgt]['weight'] += weight
                self.kg[src][tgt]['nav_weight'] += nav_weight
                existing_types = self.kg[src][tgt].get('all_types', [])
                if edge_type not in existing_types:
                    existing_types.append(edge_type)
                self.kg[src][tgt]['all_types'] = existing_types
                self.kg[src][tgt]['type'] = edge_type
                self.kg[src][tgt]['document_count'] = self.kg[src][tgt].get('document_count', 0) + len(document_sources)
            else:
                self.kg.add_edge(
                    src, tgt,
                    weight=weight,
                    nav_weight=nav_weight,
                    type=edge_type,
                    all_types=[edge_type],
                    document_count=len(document_sources)
                )

            edge_count += 1

            if edge_count % 1000 == 0:
                print(f"  processed {edge_count} edges...")

        for u, v, d in self.kg.edges(data=True):
            d['cost'] = 1.0 / (d.get('weight', 0.5) + 0.01)

        print(f"\nGraph loaded: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
        print("Relation-type distribution (Top 30):")
        for rt, count in relation_type_counter.most_common(30):
            print(f"   - {rt}: {count}")
        if parse_error_count > 0:
            print(f"WARNING: {parse_error_count} edges had no extractable relation_type")

    # =========================================================
    # Offline module-library construction
    # =========================================================
    def build_offline_module_library(self, top_percentile=0.90):
        print("\nBuilding offline comorbidity anchor module library...")
        df_s2 = pd.read_csv(self.s2_path)

        s2_graph = nx.Graph()
        threshold = df_s2['S2_Composite_Score'].quantile(top_percentile)
        df_strong_s2 = df_s2[df_s2['S2_Composite_Score'] >= threshold]

        for _, row in df_strong_s2.iterrows():
            diseases = row['Combination'].split(' + ')
            score = row['S2_Composite_Score']
            for i in range(len(diseases)):
                for j in range(i + 1, len(diseases)):
                    u, v = diseases[i], diseases[j]
                    if s2_graph.has_edge(u, v):
                        s2_graph[u][v]['weight'] += score
                    else:
                        s2_graph.add_edge(u, v, weight=score)

        communities = community.louvain_communities(s2_graph, weight='weight')

        for idx, comm in enumerate(communities):
            if len(comm) >= 2:
                shared_risks = self._extract_shared_neighbors(comm, node_type_hint='Risk')
                self.module_library.append({
                    'module_id': f"M_{idx}",
                    'diseases': list(comm),
                    'core_risks': shared_risks[:5]
                })
        print(f"Offline build complete: {len(self.module_library)} steady-state comorbidity modules.")

    def _extract_shared_neighbors(self, nodes, node_type_hint='Risk', top_k=5):
        neighbor_counts = {}
        for node in nodes:
            if node in self.kg:
                for nbr in self.kg.neighbors(node):
                    if nbr not in nodes:
                        neighbor_counts[nbr] = neighbor_counts.get(nbr, 0) + self.kg[node][nbr]['weight']
        sorted_nbrs = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)
        return [n[0] for n in sorted_nbrs][:top_k]

    # =========================================================
    # Online retrieval
    # =========================================================
    def retrieve_for_patient(self, patient_risk_vector):
        print(f"\nReceived patient risk input: {patient_risk_vector}")

        matched_module = None
        if self.ENABLE_FAST_LANE:
            matched_module = self._match_anchor_library(patient_risk_vector)
        if matched_module:
            print(f"[Fast Lane] matched steady-state module: {matched_module['module_id']}")
            return self._format_module_response(matched_module, patient_risk_vector)

        print("[Smart Lane] starting dynamic graph retrieval (patient-weighted RWR)...")
        rwr_scores = self._run_patient_weighted_rwr(patient_risk_vector)

        if not self._coherence_test(patient_risk_vector, rwr_scores):
            return {"status": "INCOHERENT",
                    "msg": "The current high-risk disease combination lacks a unified shared mechanism; manage each disease separately."}

        explanation_subgraph = self._extract_explanation_subgraph(patient_risk_vector, rwr_scores)

        print("[Smart Lane] patient-specific explanatory subgraph generated.")
        return {"status": "SUCCESS", "subgraph": explanation_subgraph}

    def _match_anchor_library(self, patient_risk_vector, overlap_threshold=0.6):
        patient_diseases = set(patient_risk_vector.keys())
        best_module = None
        best_score = 0

        for mod in self.module_library:
            mod_diseases = set(mod['diseases'])
            intersection = patient_diseases.intersection(mod_diseases)
            union = patient_diseases.union(mod_diseases)

            weight_sum = sum([patient_risk_vector.get(d, 0.5) for d in intersection])
            score = weight_sum / len(union) if union else 0

            if score > best_score and score >= overlap_threshold:
                best_score = score
                best_module = mod

        return best_module

    def _run_patient_weighted_rwr(self, patient_risk_vector, alpha=0.85, max_iter=100):
        personalization = {node: 0 for node in self.kg.nodes()}
        total_risk = sum(patient_risk_vector.values())

        if total_risk == 0:
            return personalization

        for d, risk in patient_risk_vector.items():
            if d in personalization:
                personalization[d] = risk / total_risk

        pr_scores = nx.pagerank(self.kg, alpha=alpha, personalization=personalization,
                                weight='nav_weight', max_iter=max_iter)
        return pr_scores

    def _coherence_test(self, patient_risk_vector, rwr_scores):
        top_100_nodes = sorted(rwr_scores.values(), reverse=True)[:100]
        energy_concentration = sum(top_100_nodes)
        print(f"Diffusion energy concentration (Top 100): {energy_concentration:.3f}")
        return energy_concentration > 0.4

    # =========================================================
    # Core improvement: multi-layer filtered subgraph construction
    # =========================================================
    def _extract_explanation_subgraph(
            self, patient_risk_vector, rwr_scores,
            risk_quota=12,
            prevention_quota=14,
            safety_quota=2,
            min_edge_weight=0.2,
            min_ppr_score=0.0005,
            connection_hop=3,
    ):
        """Quota scheme: keep risk-interpretation volume + force coverage of prevention advice."""
        RISK_TYPES = {
            'Risk_Factor', 'Causal', 'Predictive', 'Positive_Correlation',
            'Mechanistic', 'Mechanistic_Pathway', 'Mediating_Factor', 'Moderating_Factor'
        }
        PREVENTION_TYPES = {
            'Preventive', 'Protective_Factor', 'Screening', 'Surveillance',
            'Therapeutic', 'Diagnostic', 'Diagnostic/Therapeutic', 'Prognostic'
        }
        SAFETY_TYPES = {
            'Contraindicated', 'Adverse_Event', 'Adverse Effect', 'Adverse_Effect'
        }
        PREVENTION_SUBQUOTA = {
            'Preventive': 3, 'Protective_Factor': 2, 'Screening': 2,
            'Surveillance': 1, 'Therapeutic': 3, 'Diagnostic': 1, 'Prognostic': 1,
        }
        PREVENTION_THERAPY_KW = [
            'Diet', 'Exercise', 'Physical Activity', 'Lifestyle', 'Weight',
            'Smoking Cessation', 'Mediterranean', 'DASH', 'Education',
            'Monitoring', 'Fiber', 'Vegetable', 'Fruit', 'Sodium',
            'SGLT', 'GLP-1', 'Statin', 'ACE', 'ARB', 'Aspirin',
            'Finerenone', 'Metformin', 'Hydroxymethylglutaryl'
        ]
        EXCLUDE_KW = [
            'Adult', 'Women', 'Men', 'Children', 'Aged', 'Elderly',
            'Population', 'Demographic', 'Epidemiological'
        ]

        def bucket(et):
            if et in RISK_TYPES:
                return 'risk'
            if et in PREVENTION_TYPES:
                return 'prevention'
            if et in SAFETY_TYPES:
                return 'safety'
            return None

        def is_prev_therapy(name, et):
            if et != 'Therapeutic':
                return True
            nl = name.lower()
            return any(kw.lower() in nl for kw in PREVENTION_THERAPY_KW)

        print("Building quota-based subgraph (risk interpretation + prevention advice)...")
        patient_diseases = [d for d in patient_risk_vector if d in self.kg]
        if not patient_diseases:
            return nx.Graph()

        # How many input diseases each neighbor connects to
        nbr_diseases = {}
        for d in patient_diseases:
            for nbr in self.kg.neighbors(d):
                nbr_diseases.setdefault(nbr, set()).add(d)

        pools = {'risk': [], 'prevention': [], 'safety': []}
        seen = set()
        for d in patient_diseases:
            for nbr in self.kg.neighbors(d):
                if nbr in patient_diseases or any(kw in nbr for kw in EXCLUDE_KW):
                    continue
                e = self.kg[d][nbr]
                w, et, ppr = e.get('weight', 0.5), e.get('type', 'ASSOCIATION'), rwr_scores.get(nbr, 0.0)
                b = bucket(et)
                if b is None or w < min_edge_weight or ppr < min_ppr_score:
                    continue
                if not is_prev_therapy(nbr, et):
                    continue
                key = (nbr, et, d)
                if key in seen:
                    continue
                seen.add(key)
                n_link = len(nbr_diseases.get(nbr, {d}))
                pools[b].append({
                    'node': nbr, 'src': d, 'type': et, 'weight': w, 'ppr': ppr,
                    'n_link': n_link, 'score': n_link * 10 + ppr * 100 + w
                })

        def pick_nodes(items, quota, forbidden=None):
            """Pick nodes by shared degree first; returns an item list."""
            forbidden = forbidden or set()
            items = sorted(items, key=lambda x: (x['n_link'], x['score']), reverse=True)
            got, nodes = [], set()
            for it in items:
                if it['node'] in forbidden:
                    continue
                if it['node'] not in nodes and len(nodes) >= quota:
                    continue
                nodes.add(it['node'])
                got.append(it)
            return got

        selected = []
        used = set()

        risk_picked = pick_nodes(pools['risk'], risk_quota)
        selected += risk_picked
        used |= {x['node'] for x in risk_picked}

        # Prevention-first: subtype floor + total quota top-up
        by_type = {}
        for it in pools['prevention']:
            by_type.setdefault(it['type'], []).append(it)

        prev_picked = []
        prev_nodes = set()
        for et, q in PREVENTION_SUBQUOTA.items():
            chunk = pick_nodes(by_type.get(et, []), q, forbidden=set())
            for it in chunk:
                if it['node'] not in prev_nodes:
                    prev_nodes.add(it['node'])
                    prev_picked.append(it)
                elif it not in prev_picked:
                    prev_picked.append(it)  # same node, different edge

        need = max(0, prevention_quota - len(prev_nodes))
        if need:
            extra = pick_nodes(
                [it for it in pools['prevention'] if it['node'] not in prev_nodes],
                need
            )
            prev_picked += extra
            prev_nodes |= {x['node'] for x in extra}

        selected += prev_picked
        used |= prev_nodes

        safety_picked = pick_nodes(pools['safety'], safety_quota, forbidden=set())
        selected += safety_picked
        used |= {x['node'] for x in safety_picked}

        print(f"  risk {len({x['node'] for x in risk_picked})} | "
              f"prevention {len(prev_nodes)} | safety {len({x['node'] for x in safety_picked})}")
        print("  prevention types:", Counter(x['type'] for x in prev_picked))

        # Shared bridge (comorbidity intermediate node)
        bridges = set()
        for i, d1 in enumerate(patient_diseases):
            for d2 in patient_diseases[i + 1:]:
                try:
                    path = nx.shortest_path(self.kg, d1, d2, weight='cost', cutoff=connection_hop)
                    bridges.update(n for n in path[1:-1] if n not in patient_diseases)
                except Exception:
                    pass

        keep = set(patient_diseases) | used | bridges
        g = nx.Graph()
        allowed = RISK_TYPES | PREVENTION_TYPES | SAFETY_TYPES
        for u in keep:
            if u not in self.kg:
                continue
            for v in self.kg.neighbors(u):
                if v not in keep:
                    continue
                et = self.kg[u][v].get('type', 'ASSOCIATION')
                if et in allowed:
                    g.add_edge(u, v, **self.kg[u][v])

        if patient_diseases[0] in g:
            g = g.subgraph(nx.node_connected_component(g, patient_diseases[0])).copy()

        print(f"  final subgraph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        print("  edge types:", dict(Counter(d.get('type') for *_, d in g.edges(data=True))))
        return g

    def _format_module_response(self, module, patient_risk_vector):
        return {
            "status": "SUCCESS (FAST LANE)",
            "matched_module_id": module['module_id'],
            "core_mechanism": module['core_risks'],
            "suggested_interpretation": f"Your risk combination matches the classic comorbidity pattern {module['diseases']}, with mechanisms pointing to pathways such as {module['core_risks'][0]}."
        }

    # =========================================================
    # Visualization tools
    # =========================================================
    def visualize_subgraph(self, subgraph, patient_risk_vector, save_path=None):
        if not subgraph.nodes():
            print("WARNING: subgraph is empty; cannot visualize")
            return

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)

        patient_nodes = [n for n in subgraph.nodes() if n in patient_risk_vector]
        other_nodes = [n for n in subgraph.nodes() if n not in patient_risk_vector]

        nx.draw_networkx_nodes(subgraph, pos, nodelist=patient_nodes,
                               node_color='#FF6B6B', node_size=2000,
                               label='patient disease', alpha=0.9, edgecolors='black', linewidths=2)

        nx.draw_networkx_nodes(subgraph, pos, nodelist=other_nodes,
                               node_color='#4ECDC4', node_size=1200,
                               label='risk/mechanism/treatment', alpha=0.9)

        edge_types = {}
        for u, v, d in subgraph.edges(data=True):
            etype = d.get('type', 'ASSOCIATION')
            if etype not in edge_types:
                edge_types[etype] = []
            edge_types[etype].append((u, v))

        color_map = {
            'Risk_Factor': '#FF9F43',
            'Causal': '#FF6B6B',
            'Therapeutic': '#1DD1A1',
            'Preventive': '#5F27CD',
            'Protective_Factor': '#10AC84',
            'Prognostic': '#54A0FF',
            'Diagnostic': '#FFD93D',
            'Associational': '#C8D6E5'
        }

        for etype, edges in edge_types.items():
            nx.draw_networkx_edges(subgraph, pos, edgelist=edges,
                                   edge_color=color_map.get(etype, '#95A5A6'),
                                   width=2.0, alpha=0.7, label=etype)

        labels = {n: n[:25] + '...' if len(n) > 25 else n for n in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=7, font_weight='bold')

        plt.title(f"Comorbidity explanatory subgraph (nodes={subgraph.number_of_nodes()}, edges={subgraph.number_of_edges()})",
                  fontsize=14, pad=20)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Subgraph saved to: {save_path}")

        #plt.show()

    def print_subgraph_summary(self, subgraph, patient_risk_vector):
        print("\n" + "=" * 70)
        print("Retrieval subgraph structure summary")
        print("=" * 70)

        print(f"Total nodes: {subgraph.number_of_nodes()}")
        print(f"Total edges: {subgraph.number_of_edges()}")
        print(f"Graph density: {nx.density(subgraph):.3f}")
        print(f"Connected components: {nx.number_connected_components(subgraph)}")

        patient_nodes = [n for n in subgraph.nodes() if n in patient_risk_vector]
        mechanism_nodes = [n for n in subgraph.nodes() if n not in patient_risk_vector]

        print(f"\nPatient disease nodes ({len(patient_nodes)}):")
        for n in patient_nodes:
            print(f"   - {n} (risk={patient_risk_vector.get(n, 'N/A')})")

        print(f"\nMechanism/risk/treatment nodes ({len(mechanism_nodes)}):")
        # Show Top 20 sorted by degree
        sorted_mechanism = sorted(mechanism_nodes, key=lambda x: subgraph.degree(x), reverse=True)[:20]
        for n in sorted_mechanism:
            degree = subgraph.degree(n)
            edge_types = set()
            for nbr in subgraph.neighbors(n):
                if subgraph.has_edge(n, nbr):
                    etype = subgraph[n][nbr].get('type', 'UNKNOWN')
                    edge_types.add(etype)
            print(f"   - {n} (degree={degree}, relation types={edge_types})")

        edge_types = {}
        for u, v, d in subgraph.edges(data=True):
            etype = d.get('type', 'ASSOCIATION')
            edge_types[etype] = edge_types.get(etype, 0) + 1

        print("\nEdge-type distribution:")
        for etype, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {etype}: {count}")

        if len(patient_nodes) >= 2:
            print("\nShortest paths between patient diseases:")
            for i, n1 in enumerate(patient_nodes):
                for n2 in patient_nodes[i + 1:]:
                    try:
                        path = nx.shortest_path(subgraph, n1, n2)
                        print(
                            f"   - {n1[:25]} <-> {n2[:25]}: length={len(path) - 1}, path={' -> '.join([p[:20] for p in path])}")
                    except:
                        print(f"   - {n1[:25]} <-> {n2[:25]}: no path")

        print("=" * 70 + "\n")

    def generate_clinical_report(self, subgraph, patient_risk_vector):
        """Generate the clinical interpretation report"""
        print("\n" + "=" * 70)
        print("Clinical interpretation report")
        print("=" * 70)

        patient_nodes = [n for n in subgraph.nodes() if n in patient_risk_vector]
        mechanism_nodes = [n for n in subgraph.nodes() if n not in patient_risk_vector]

        # 1. Risk factors
        risk_factors = []
        for node in mechanism_nodes:
            for nbr in subgraph.neighbors(node):
                if nbr in patient_nodes:
                    edge_type = subgraph[node][nbr].get('type', '')
                    if any(k in edge_type.upper() for k in ['RISK', 'PREDICT', 'CAUSE']):
                        risk_factors.append(node)

        if risk_factors:
            print("\nShared risk factors:")
            for rf in set(risk_factors):
                print(f"   - {rf}")

        # 2. Treatment recommendations
        treatments = []
        for node in mechanism_nodes:
            for nbr in subgraph.neighbors(node):
                if nbr in patient_nodes:
                    edge_type = subgraph[node][nbr].get('type', '')
                    if any(k in edge_type.upper() for k in ['TREAT', 'THERAPY', 'PREVENT']):
                        treatments.append(node)

        if treatments:
            print("\nPotential therapeutic interventions:")
            for t in set(treatments):
                print(f"   - {t}")

        # 3. Comorbidity mechanisms
        print("\nComorbidity association mechanisms:")
        for i, n1 in enumerate(patient_nodes):
            for n2 in patient_nodes[i + 1:]:
                try:
                    path = nx.shortest_path(subgraph, n1, n2)
                    if len(path) > 2:
                        intermediates = path[1:-1]
                        print(f"   - {n1[:25]} <-> {n2[:25]}: via {intermediates}")
                except:
                    pass

        print("=" * 70 + "\n")

    def save_subgraph_complete(self, subgraph, patient_risk_vector, save_dir, eid):
        """
        Save the full subgraph data (image + graph structure + full edge info)
        """
        import os
        import json

        os.makedirs(save_dir, exist_ok=True)
        base_path = os.path.join(save_dir, f"eid_{eid}")

        # 1. Save visualization images
        try:
            self.visualize_subgraph(subgraph, patient_risk_vector, save_path=f"{base_path}.png")
        except Exception as e:
            print(f"  WARNING: visualization save failed: {e}")

        # 2. Clean data: convert list/dict to string (GraphML compatible)
        clean_graph = subgraph.copy()
        for u, v, data in clean_graph.edges(data=True):
            for key, value in list(data.items()):
                if isinstance(value, (list, dict, set)):
                    data[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, (np.floating, np.integer)):
                    data[key] = float(value)

        for node, data in clean_graph.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, (list, dict, set)):
                    data[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, (np.floating, np.integer)):
                    data[key] = float(value)

        # 3. Save graph structure (GraphML)
        try:
            nx.write_graphml(clean_graph, f"{base_path}.graphml")
        except Exception as e:
            print(f"  WARNING: GraphML save failed: {e}")

        # 3. Save full edge data (retrieve complete info from the raw KG)
        edge_details = []
        for u, v, data in subgraph.edges(data=True):
            # Look up the full document_sources of this edge in the raw graph
            full_edge_data = self._get_full_edge_data(u, v)
            edge_details.append({
                'source_term': u,
                'target_term': v,
                'subgraph_edge_data': data,  # edge attributes in the subgraph (weight, type, etc.)
                'full_kg_edge_data': full_edge_data  # complete data from the raw KG
            })

        # 4. Save subgraph metadata
        metadata = {
            'eid': eid,
            'patient_risk_vector': patient_risk_vector,
            'node_count': subgraph.number_of_nodes(),
            'edge_count': subgraph.number_of_edges(),
            'edge_details': edge_details
        }

        with open(f"{base_path}_edges.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return f"{base_path}_edges.json"

    def _get_full_edge_data(self, src, tgt):
        """
        Fetch full edge information (including document_sources) from the raw KG
        """
        # Save raw edges data during initialization
        if hasattr(self, '_original_edges'):
            for edge_id, edge_data in self._original_edges.items():
                e_src = self.mapping_dict.get(edge_data.get('source_term', ''), edge_data.get('source_term', ''))
                e_tgt = self.mapping_dict.get(edge_data.get('target_term', ''), edge_data.get('target_term', ''))
                if (e_src == src and e_tgt == tgt) or (e_src == tgt and e_tgt == src):
                    return edge_data
        return None
