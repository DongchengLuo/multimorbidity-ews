/* ============================================================================
 * json-render report renderer (Vue 3, no build step)
 * ----------------------------------------------------------------------------
 * Input: report_<app_no>.json ({root, elements})
 * Components: Page / Section / SubSection / Paragraph / RiskDashboard / Gauge /
 *       Metric / RiskBadge / Callout / List / Table / KeyValue / Timeline /
 *       ActionItem / Evidence / Icon / Divider / Columns / Card
 * Visual hierarchy: high-risk gauges are largest and most prominent, medium next, low de-emphasized.
 * ========================================================================== */
(function () {
  const { reactive } = Vue;

  const jrState = reactive({ spec: null });

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  // **bold** -> <strong>
  function fmt(text) {
    return esc(text).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  }

  // Icons: prefer report-level SVG (ui_icons/report_ui), else fall back to generic icons
  function iconUrl(name) {
    if (!name) return '';
    if (name.indexOf('/') >= 0 || name.indexOf('.') >= 0) return '/ui_icons/' + name;
    return '/ui_icons/report_ui/' + name + '.svg';
  }

  const TIER_FILL = { 4: 0.94, 3: 0.78, 2: 0.52, 1: 0.26, 0: 0.1 };
  const TIER_CLASS = { 4: 't4', 3: 't3', 2: 't2', 1: 't1', 0: 't0' };
  const COLOR_HEX = { red: '#e5484d', orange: '#f5a524', green: '#30a46c', blue: '#2b85e4' };

  /* ---------------- Base components ---------------- */
  const JrPage = {
    props: ['el'],
    template: `
      <div class="jr-page">
        <header class="jr-page-head">
          <div class="jr-page-title">{{ el.props.title }}</div>
          <div class="jr-page-sub" v-if="el.props.subtitle">{{ el.props.subtitle }}</div>
          <div class="jr-page-meta">
            <span v-if="el.props.patient">{{ el.props.patient }}</span>
            <span v-if="el.props.generatedAt">Generated: {{ el.props.generatedAt }}</span>
          </div>
        </header>
        <slot></slot>
      </div>`
  };

  const JrSection = {
    props: ['el'],
    methods: { iconUrl },
    template: `
      <section class="jr-section">
        <h2 class="jr-section-title">
          <span class="jr-section-bar"></span>
          <img v-if="el.props.icon" class="jr-section-icon" :src="iconUrl(el.props.icon)" alt="">
          {{ el.props.title }}
        </h2>
        <slot></slot>
      </section>`
  };

  const JrSubSection = {
    props: ['el'],
    methods: { iconUrl },
    template: `
      <div class="jr-subsection">
        <h3 class="jr-subsection-title">
          <img v-if="el.props.icon" class="jr-subsection-icon" :src="iconUrl(el.props.icon)" alt="">
          {{ el.props.title }}
        </h3>
        <slot></slot>
      </div>`
  };

  const JrParagraph = {
    props: ['el'],
    computed: { html() { return fmt(this.el.props.text || ''); } },
    template: `<p class="jr-p" v-html="html"></p>`
  };

  const JrCallout = {
    props: ['el'],
    computed: {
      cls() { return 'jr-callout jr-callout-' + (this.el.props.type || 'info'); },
      html() { return fmt(this.el.props.text || ''); }
    },
    template: `
      <div :class="cls">
        <div class="jr-callout-title" v-if="el.props.title">{{ el.props.title }}</div>
        <div class="jr-callout-text" v-html="html"></div>
      </div>`
  };

  const JrList = {
    props: ['el'],
    computed: {
      items() { return this.el.props.items || []; },
      tag() { return this.el.props.ordered ? 'ol' : 'ul'; }
    },
    methods: { fmt },
    template: `
      <component :is="tag" class="jr-list">
        <li v-for="(it, i) in items" :key="i" v-html="fmt(it)"></li>
      </component>`
  };

  const JrTable = {
    props: ['el'],
    methods: { fmt },
    template: `
      <div class="jr-table-wrap">
        <table class="jr-table">
          <thead><tr><th v-for="(h,i) in (el.props.headers||[])" :key="i">{{ h }}</th></tr></thead>
          <tbody>
            <tr v-for="(row,ri) in (el.props.rows||[])" :key="ri">
              <td v-for="(c,ci) in row" :key="ci" v-html="fmt(c)"></td>
            </tr>
          </tbody>
        </table>
      </div>`
  };

  const JrKeyValue = {
    props: ['el'],
    methods: { fmt },
    template: `
      <div class="jr-kv">
        <div class="jr-kv-row" v-for="(it,i) in (el.props.items||[])" :key="i">
          <span class="jr-kv-key">{{ it.key }}</span>
          <span class="jr-kv-val" v-html="fmt(it.value)"></span>
        </div>
      </div>`
  };

  const JrTimeline = {
    props: ['el'],
    methods: { fmt },
    template: `
      <div class="jr-timeline">
        <div class="jr-tl-item" v-for="(it,i) in (el.props.items||[])" :key="i">
          <div class="jr-tl-dot"></div>
          <div class="jr-tl-body">
            <div class="jr-tl-time">{{ it.time }}</div>
            <div class="jr-tl-title">{{ it.title }}</div>
            <div class="jr-tl-text" v-html="fmt(it.text)"></div>
          </div>
        </div>
      </div>`
  };

  const JrActionItem = {
    props: ['el'],
    computed: {
      cls() {
        const p = String(this.el.props.priority || 'Medium');
        const pl = p.toLowerCase();
        if (pl.indexOf('high') >= 0) return 'p-high';
        if (pl.indexOf('low') >= 0) return 'p-low';
        return 'p-mid';
      }
    },
    methods: { fmt },
    template: `
      <div class="jr-action" :class="cls">
        <div class="jr-action-head">
          <span class="jr-action-pri">{{ el.props.priority || 'Medium' }}</span>
          <span class="jr-action-title">{{ el.props.title }}</span>
        </div>
        <div class="jr-action-detail" v-html="fmt(el.props.detail)"></div>
        <div class="jr-action-tags" v-if="el.props.tags && el.props.tags.length">
          <span class="jr-tag" v-for="(t,i) in el.props.tags" :key="i">{{ t }}</span>
        </div>
      </div>`
  };

  const JrEvidence = {
    props: ['el'],
    methods: { fmt },
    template: `
      <div class="jr-evidence">
        <span class="jr-evidence-icon">◈</span>
        <span class="jr-evidence-text" v-html="fmt(el.props.text)"></span>
        <span class="jr-evidence-src" v-if="el.props.source">{{ el.props.source }}</span>
      </div>`
  };

  const JrRiskBadge = {
    props: ['el'],
    computed: {
      cls() {
        const l = String(this.el.props.level || '');
        const ll = l.toLowerCase();
        if (ll.indexOf('very high') >= 0) return 'b-t4';
        if (ll.indexOf('high') >= 0) return 'b-t3';
        if (ll.indexOf('medium') >= 0 || ll.indexOf('moderate') >= 0) return 'b-t2';
        return 'b-t1';
      }
    },
    template: `<span class="jr-badge" :class="cls">{{ el.props.text || el.props.level }}</span>`
  };

  const JrMetric = {
    props: ['el'],
    computed: {
      cls() {
        const t = this.el.props.tier || 0;
        return 'jr-metric ' + (TIER_CLASS[t] || 't0');
      }
    },
    template: `
      <div :class="cls">
        <div class="jr-metric-label">{{ el.props.label }}</div>
        <div class="jr-metric-value">{{ el.props.value }}<span class="jr-metric-unit" v-if="el.props.unit">{{ el.props.unit }}</span></div>
        <div class="jr-metric-hint" v-if="el.props.hint">{{ el.props.hint }}</div>
      </div>`
  };

  const JrIcon = {
    props: ['el'],
    computed: {
      color() { return COLOR_HEX[this.el.props.color] || COLOR_HEX.blue; },
      size() { return this.el.props.size === 'lg' ? 28 : (this.el.props.size === 'sm' ? 16 : 20); }
    },
    template: `
      <svg :width="size" :height="size" viewBox="0 0 24 24" class="jr-icon">
        <circle cx="12" cy="12" r="9" :fill="color" opacity="0.15"/>
        <circle cx="12" cy="12" r="4" :fill="color"/>
      </svg>`
  };

  const JrDivider = { props: ['el'], template: `<hr class="jr-divider">` };

  const JrColumns = {
    props: ['el'],
    computed: {
      style() {
        const r = String(this.el.props.ratio || '1:1').split(':');
        return { display: 'grid', gridTemplateColumns: r.map(x => (x || 1) + 'fr').join(' '), gap: '16px' };
      }
    },
    template: `<div :style="style"><slot></slot></div>`
  };

  const JrCard = {
    props: ['el'],
    template: `
      <div class="jr-card">
        <div class="jr-card-title" v-if="el.props.title">{{ el.props.title }}</div>
        <slot></slot>
      </div>`
  };

  /* ---------------- Dashboard ---------------- */
  const JrGauge = {
    props: ['el'],
    computed: {
      tier() { return Number(this.el.props.tier || 0); },
      pos() {
        if (this.el.props.pos != null) return Math.max(0, Math.min(100, Number(this.el.props.pos)));
        if (this.el.props.value != null && this.el.props.max) {
          return Math.max(0, Math.min(100, 100 * Number(this.el.props.value) / Number(this.el.props.max)));
        }
        return (TIER_FILL[this.tier] != null ? TIER_FILL[this.tier] : 0.1) * 100;
      },
      sizeClass() { return 'g-' + (this.el.props.size || 'md'); },
      needleStyle() {
        const angle = -90 + 360 * this.pos / 100;
        return { transform: 'rotate(' + angle + 'deg)', transformOrigin: '50px 50px' };
      },
      tierText() { return this.el.props.tierLabel || ''; }
    },
    template: `
      <div class="jr-gauge" :class="sizeClass">
        <svg viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="42" class="jr-zone zone-low"/>
          <circle cx="50" cy="50" r="42" class="jr-zone zone-mid"/>
          <circle cx="50" cy="50" r="42" class="jr-zone zone-high"/>
          <line x1="50" y1="50" x2="50" y2="16" class="jr-needle" :style="needleStyle"/>
          <circle cx="50" cy="50" r="3.2" class="jr-needle-hub"/>
          <text x="50" y="74" class="jr-gauge-tier">{{ tierText }}</text>
        </svg>
      </div>`
  };

  const JrRiskDashboard = {
    props: ['el'],
    methods: { iconUrl },
    computed: {
      items() {
        const arr = (this.el.props.items || []).slice();
        arr.sort((a, b) => (b.tier || 0) - (a.tier || 0));
        return arr;
      },
      groups() {
        const g = { high: [], mid: [], low: [] };
        this.items.forEach(it => {
          const t = Number(it.tier || 0);
          if (t >= 3) g.high.push(it);
          else if (t === 2) g.mid.push(it);
          else g.low.push(it);
        });
        return g;
      }
    },
    template: `
      <div class="jr-dashboard">
        <div class="jr-dash-title" v-if="el.props.title">{{ el.props.title }}</div>

        <div class="jr-dash-group" v-if="groups.high.length">
          <div class="jr-dash-group-label high">Priority Focus · High Risk</div>
          <div class="jr-dash-grid high">
            <div class="jr-dash-item high" v-for="(it,i) in groups.high" :key="'h'+i">
              <img class="jr-dash-icon" :src="iconUrl(it.icon)" alt="">
              <jr-gauge :el="{ props: { tier: it.tier, pos: it.pos, tierLabel: it.tierLabel, size: 'lg' } }"></jr-gauge>
              <div class="jr-dash-name">{{ it.disease }}</div>
              <div class="jr-dash-tier high">{{ it.tierLabel }}</div>
              <div class="jr-dash-models" v-if="it.models && it.models.length">
                <span v-for="(m,mi) in it.models.slice(0,3)" :key="mi">{{ m.name }} {{ m.score }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="jr-dash-group" v-if="groups.mid.length">
          <div class="jr-dash-group-label mid">Ongoing Monitoring · Medium Risk</div>
          <div class="jr-dash-grid mid">
            <div class="jr-dash-item mid" v-for="(it,i) in groups.mid" :key="'m'+i">
              <img class="jr-dash-icon sm" :src="iconUrl(it.icon)" alt="">
              <jr-gauge :el="{ props: { tier: it.tier, pos: it.pos, tierLabel: it.tierLabel, size: 'md' } }"></jr-gauge>
              <div class="jr-dash-name">{{ it.disease }}</div>
              <div class="jr-dash-tier mid">{{ it.tierLabel }}</div>
              <div class="jr-dash-models" v-if="it.models && it.models.length">
                <span v-for="(m,mi) in it.models.slice(0,2)" :key="mi">{{ m.name }} {{ m.score }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="jr-dash-footnote">
          Threshold note: each model has its own raw threshold; values are mapped to 0–100 here (Low 0–33 / Medium 33–66 / High 66–100);
          The needle position indicates the risk band for that disease; see the raw score below each gauge.
        </div>

        <div class="jr-dash-group" v-if="groups.low.length">
          <div class="jr-dash-group-label low">Routine Monitoring · Low Risk</div>
          <div class="jr-dash-grid low">
            <div class="jr-dash-item low" v-for="(it,i) in groups.low" :key="'l'+i">
              <jr-gauge :el="{ props: { tier: it.tier, pos: it.pos, tierLabel: '', size: 'sm' } }"></jr-gauge>
              <div class="jr-dash-name">{{ it.disease }}</div>
            </div>
          </div>
        </div>
      </div>`
  };

  /* ---------------- Recursive nodes ---------------- */
  const JrNode = {
    name: 'JrNode',
    props: { id: String },
    inject: ['jrState'],
    computed: {
      el() {
        const spec = this.jrState.spec;
        return (spec && spec.elements && spec.elements[this.id]) || { type: 'Paragraph', props: { text: '' } };
      }
    },
    template: `
      <component :is="'jr-' + String(el.type || 'Paragraph').toLowerCase()" :el="el">
        <jr-node v-for="c in (el.children || [])" :key="c" :id="c"></jr-node>
      </component>`
  };

  window.JR = {
    state: jrState,
    components: {
      'jr-node': JrNode,
      'jr-page': JrPage,
      'jr-section': JrSection,
      'jr-subsection': JrSubSection,
      'jr-paragraph': JrParagraph,
      'jr-callout': JrCallout,
      'jr-list': JrList,
      'jr-table': JrTable,
      'jr-keyvalue': JrKeyValue,
      'jr-timeline': JrTimeline,
      'jr-actionitem': JrActionItem,
      'jr-evidence': JrEvidence,
      'jr-riskbadge': JrRiskBadge,
      'jr-metric': JrMetric,
      'jr-icon': JrIcon,
      'jr-divider': JrDivider,
      'jr-columns': JrColumns,
      'jr-card': JrCard,
      'jr-gauge': JrGauge,
      'jr-riskdashboard': JrRiskDashboard
    }
  };
})();
