import re

# Define your ticket codes for each phase
MVP_TICKETS = {
    "AIM2-001",
    "AIM2-002",
    "AIM2-003",
    "AIM2-004",
    "AIM2-005",
    "AIM2-011",
    "AIM2-012",
    "AIM2-013",
    "AIM2-031",
    "AIM2-035",
    "AIM2-036",
}
# All other tickets will be considered Post-MVP


def parse_checklist(file_path):
    tasks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"- \[( |x)\] \*\*(AIM2-\d{3})-(\d{2})\*\* (.+)", line)
            if match:
                checked = match.group(1) == "x"
                ticket = match.group(2)
                subcode = match.group(3)
                text = match.group(4)
                tasks.append(
                    {
                        "checked": checked,
                        "ticket": ticket,
                        "subcode": subcode,
                        "text": text,
                        "raw": line.rstrip(),
                    }
                )
    return tasks


def group_tasks(tasks):
    mvp = []
    post_mvp = []
    for task in tasks:
        if task["ticket"] in MVP_TICKETS:
            mvp.append(task)
        else:
            post_mvp.append(task)
    return mvp, post_mvp


def write_grouped_checklist(mvp, post_mvp, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "# AIM2 Project Checklist Grouped by MVP and Post-MVP Phases\n\n---\n\n"
        )
        f.write("## MVP Phase Tasks\n\n")
        for task in mvp:
            f.write(f"{task['raw']}\n")
        f.write("\n---\n\n")
        f.write("## Post-MVP Phase Tasks\n\n")
        for task in post_mvp:
            f.write(f"{task['raw']}\n")


if __name__ == "__main__":
    tasks = parse_checklist("docs/checklist.md")
    mvp, post_mvp = group_tasks(tasks)
    print(f"MVP tasks: {len(mvp)}")
    print(f"Post-MVP tasks: {len(post_mvp)}")
    print(f"Total tasks: {len(tasks)}")
    write_grouped_checklist(mvp, post_mvp, "docs/checklist_grouped_by_MVP_full.md")
    print("Grouped checklist written to docs/checklist_grouped_by_MVP_full.md")
