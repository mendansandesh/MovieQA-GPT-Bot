from youtube_transcript_api import YouTubeTranscriptApi
import re

video_id = "dQw4w9WgXcQ"

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.list("UYhKDweME3A")
    transcript = transcript_list.find_transcript(['en'])
    fetched = transcript.fetch()
    raw_text = " ".join([s.text for s in fetched.snippets])

    text = re.sub(r"\[.*?\]", "", raw_text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)
    text = re.sub(r"\s+", " ", text)
    print(text.strip())       
except Exception as e:
    print(f"❌ Error occurred: {e}")
