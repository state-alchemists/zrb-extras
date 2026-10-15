"""Maintainer tasks: the LLM code review CI runs on every pull request.

Copied from zrb's own `zrb_init.py`. `src/zrb_init.py` is the user-facing
example; this one is for working on zrb-extras itself.
"""

import json
import os
from urllib.parse import urlparse

import requests

from zrb import AnyContext, CmdTask, LLMTask, StrInput, Tpl, cli, make_task
from zrb.llm.tool import read_file as read_file_tool
from zrb.llm.tool import run_shell_command as run_shell_command_tool
from zrb.util.file import read_file

git_diff = CmdTask(
    name="git-diff",
    input=StrInput(
        name="range",
        description="Git range to review (e.g. origin/main...HEAD)",
        prompt="Git range",
        default="origin/main...HEAD",
    ),
    # --name-status, not --stat: stat elides long paths with "...".
    cmd=Tpl('git diff --name-status "{ctx.input.range}"'),
)

review_code = cli.add_task(
    LLMTask(
        name="review-code",
        description="🔍 Review changed code with the LLM reviewer",
        input=[
            StrInput(
                name="range",
                description="Git range to review (e.g. origin/main...HEAD)",
                prompt="Git range",
                default="origin/main...HEAD",
            ),
        ],
        tools=[read_file_tool, run_shell_command_tool],
        yolo=True,
        retries=0,
        message=Tpl(
            "Review the changes in git range {ctx.input.range}."
            " Changed files (status, path): {ctx.xcom['git-diff'].peek()}"
            " Read the diff with git and read the files it touches,"
            " including the callers of whatever the diff changes."
            " Report only defects the diff introduces or makes worse:"
            " a wrong result, a crash, a leak, a security hole."
            " Skip style, speculation, pre-existing behaviour, and limits"
            " the code or docs already state that normal use cannot trigger."
            " Every finding needs a concrete trigger: the input or state,"
            " the path from normal use that reaches the code, and the wrong"
            " outcome. If you cannot name all three, drop the finding."
            " Check a claim before you report it: you have a shell, so run"
            " a small `python -c` or git command to confirm a behaviour you"
            " are unsure of, such as whether a regex matches or a function"
            " returns what you assume."
            " Do NOT run the test suite, build, or any linter --"
            " CI already ran them in a separate step and a second run"
            " here only burns minutes."
            " Work directly: no skill activation, no delegation,"
            " no plan -- go straight to reading the diff."
            " Rank each finding Blocker (data loss, a crash, a wrong result"
            " in normal use, a security hole), Major (a wrong result in a"
            " realistic but uncommon case) or Minor. Report at most five,"
            " most severe first."
            " Your review should be in markdown format."
            " Give every finding its own section with Severity, Problem,"
            " Trigger, Location (file:line) and Suggestion, and end the"
            " report with a verdict line reading 'LGTM' when there is no"
            " Blocker or Major finding (Minor ones may still be listed),"
            " otherwise 'Request changes'."
        ),
    ),
    alias="review",
)


@make_task(
    name="submit-comment",
    description="💬 Post code review comment",
    retries=2,
    group=cli,
)
def submit_comment(ctx: AnyContext):
    """Post the review report as a comment on the PR that triggered this run.

    The PR number comes from GITHUB_EVENT_PATH, the event payload GitHub writes
    for every run, so this takes no argument.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    missing = [
        name
        for name, value in (
            ("GITHUB_TOKEN", token),
            ("GITHUB_EVENT_PATH", event_path),
            ("GITHUB_REPOSITORY", repository),
        )
        if value == ""
    ]
    if missing:
        raise ValueError(
            f"Not running inside GitHub Actions? Missing: {', '.join(missing)}"
        )
    body: str = ctx.xcom.get("review-code").peek()
    pr_number = json.loads(read_file(event_path)).get("pull_request", {}).get("number")
    if pr_number is None:
        raise ValueError("The event payload carries no pull_request.number")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    if urlparse(api_url).hostname != "api.github.com":
        raise ValueError(f"Refusing to send GITHUB_TOKEN to {api_url}")
    response = requests.post(
        f"{api_url}/repos/{repository}/issues/{pr_number}/comments",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"body": body},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Could not comment on PR #{pr_number}: "
            f"{response.status_code} {response.text}"
        )
    ctx.print(f"Posted {len(body)} bytes to PR #{pr_number}")


_ = git_diff >> review_code >> submit_comment
