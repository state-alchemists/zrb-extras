from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from zrb import AnyContext


def fetch_youtube_transcript(ctx: AnyContext, url: str):
    """
    Get transcript of a Youtube video
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract a video id from the provided string/url")
    ctx.print(f"VIDEO ID: {video_id}", plain=True)
    try:
        # get_transcript / get_transcripts / fetch — either is fine; common call:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        return " ".join([snippet.text for snippet in transcript.snippets])
    except TranscriptsDisabled:
        ctx.log_error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        ctx.log_error("No transcript found for this video.")
    except VideoUnavailable:
        ctx.log_error("Video unavailable.")
    except Exception as e:
        # RequestBlocked / IpBlocked etc can happen in cloud environments
        ctx.log_error("Error fetching transcript:", e)
    return None


def extract_video_id(url: str) -> str | None:
    url = url.strip()
    # raw id
    if "://" not in url and "=" not in url and "/" not in url:
        return url
    parsed = urlparse(url)
    # /shorts/VIDEOID or /shorts/VIDEOID/...
    if parsed.path.startswith("/shorts/"):
        parts = parsed.path.split("/")
        # /shorts/ID  -> parts ['', 'shorts', 'ID']
        if len(parts) >= 3 and parts[2]:
            return parts[2]
    # youtu.be/ID
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.lstrip("/")
    # /watch?v=ID or &v=ID
    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]
    # embed path: /embed/ID
    if parsed.path.startswith("/embed/"):
        return parsed.path.split("/")[2]
    return None
