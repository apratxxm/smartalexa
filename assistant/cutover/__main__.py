import argparse
import json

from .backup import legacy_startup_check, make_backup, restore_to, verify_backup


def main():
    parser = argparse.ArgumentParser(description="Offline cutover backup tools")
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--root", required=True)
    backup.add_argument("--zip", required=True)
    backup.add_argument("--manifest", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--zip", required=True)
    verify.add_argument("--manifest", required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--zip", required=True)
    restore.add_argument("--scratch", required=True)
    legacy = commands.add_parser("legacy-check")
    legacy.add_argument("--restored-root", required=True)
    legacy.add_argument("--python", required=True)
    legacy.add_argument("--live-root", required=True)
    args = parser.parse_args()
    if args.command == "backup":
        result = make_backup(args.root, args.zip, manifest_path=args.manifest)
    elif args.command == "verify":
        result = verify_backup(args.zip, args.manifest)
    elif args.command == "restore":
        result = restore_to(args.zip, args.scratch)
    else:
        result = legacy_startup_check(args.restored_root, args.python, live_root=args.live_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
