# cli.py (full version)
import sys
import json
import os
from batch_processor import BatchProcessor


def read_from_file(file_path):
    """Read app_no values from a file"""
    app_nos = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip blank lines and comments
                app_nos.append(line)
    return app_nos


def interactive_input():
    """Interactive input"""
    print("Medical Risk Assessment System - Interactive Input")
    print("=" * 50)
    print("Enter app_no values, one per line; finish with a blank line:")

    app_nos = []
    while True:
        try:
            line = input(f"app_no [{len(app_nos) + 1}]: ").strip()
            if not line:
                if len(app_nos) > 0:
                    break
                else:
                    print("Enter at least one app_no")
                    continue
            app_nos.append(line)
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nInput interrupted")
            return None

    return app_nos


def print_usage():
    print("Medical Risk Assessment CLI")
    print("=" * 50)
    print("Usage:")
    print("  1. Pass app_no directly:")
    print("     python cli.py 90000001 90000002")
    print()
    print("  2. Read from a file:")
    print("     python cli.py --file app_nos.txt")
    print()
    print("  3. Interactive input:")
    print("     python cli.py --interactive")
    print()
    print("  4. Use a config file:")
    print("     python cli.py --config config.json")
    print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    app_nos = []

    # Parse command-line arguments
    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Error: please specify an input file")
            sys.exit(1)
        app_nos = read_from_file(sys.argv[2])

    elif sys.argv[1] == "--interactive":
        app_nos = interactive_input()
        if app_nos is None:
            return

    elif sys.argv[1] == "--config":
        if len(sys.argv) < 3:
            print("Error: please specify a config file")
            sys.exit(1)
        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            config = json.load(f)
            app_nos = config.get("app_nos", [])

    else:
        # Direct app_no input
        app_nos = sys.argv[1:]
        invalid = [a for a in app_nos if a.startswith("-")]
        if invalid:
            print(f"Error: unrecognized argument {invalid[0]} (app_no must not start with -)")
            print()
            print_usage()
            sys.exit(1)

    if not app_nos:
        print("Error: no valid app_no provided")
        sys.exit(1)

    print(f"\nStarting batch with {len(app_nos)} app_no")
    print(f"app_nos: {app_nos}")

    # Process batch
    processor = BatchProcessor()
    results = processor.process_batch(app_nos)

    # Output results
    print("\n" + "=" * 60)
    print("Assessment results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Save to file
    output_file = f"batch_results_{len(app_nos)}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
