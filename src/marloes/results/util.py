def get_latest_uid(dir: str) -> int:
    with open(f"{dir}/uid.txt", "r") as f:
        return int(f.read().strip()) - 2
