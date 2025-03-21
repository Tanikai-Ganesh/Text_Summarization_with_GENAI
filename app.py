import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain.docstore.document import Document


## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')



## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

# Replace with your YouTube Data API key
with st.sidebar:
    YOUTUBE_API_KEY=st.text_input("Youtube API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

# Function to extract video ID from URL
def get_video_id(youtube_url):
    video_id_match = re.search(r"(?:v=|v/|youtu.be/)([a-zA-Z0-9_-]{11})", youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

# Function to fetch video metadata using YouTube Data API
def get_video_metadata(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()
    
    if response["items"]:
        video_info = response["items"][0]["snippet"]
        return {
            "title": video_info["title"],
            "description": video_info["description"],
            "channel": video_info["channelTitle"],
            "published_at": video_info["publishedAt"]
        }
    else:
        return None

# Function to fetch video transcript
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript])
        return full_text
    except Exception as e:
        return f"Transcript unavailable: {str(e)}"

# Main function to load YouTube content
def load_youtube_content(youtube_url):
    try:
        # Extract video ID
        video_id = get_video_id(youtube_url)
        
        # Fetch metadata
        metadata = get_video_metadata(video_id)
        if not metadata:
            return "Video metadata not found."
        
        # Fetch transcript
        transcript = get_video_transcript(video_id)
        
        # Combine results
        result = {
            "metadata": metadata,
            "transcript": transcript
        }
        return result
    
    except Exception as e:
        return f"Error: {str(e)}"



## Gemma Model USsing Groq API
llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    content = load_youtube_content(generic_url)
                    docs = [Document(page_content=content['transcript'],metadata={"source":"local"})]
                    print(docs)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()
                    print(docs)
                
                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
                    