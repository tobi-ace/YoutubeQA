import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi


def main():
    st.set_page_config(layout="wide", page_title= "YoutubeQA")
    st.title("YoutubeQA")
    st.caption("Ask an LLM about what was said in a Youtube video")

    col1, col2 = st.columns(2)

    container_height = 500

    url = None
    with col1:
        st.subheader("Video")
        with st.container(height=container_height):
            url = st.text_input(label="URL", placeholder="Enter a Youtube video url")
            if url:
                timestamp = int(st.session_state.time_stamp) if "timestamp" in st.session_state else 0
                with st.empty():
                        st.video(url, start_time=timestamp )

    

    with col2:
        st.subheader("LLM")
        messages = st.container(height=container_height)

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role":"assistant", "content":"What would you like to know from this video"}]

    for message in st.session_state.messages:
        messages.chat_message(message["role"]).write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role":"user", "content":prompt})
        messages.chat_message("user").write(prompt)

    if prompt :
        if url:
            transcript, transcript_text_list = download_transcript(url)
            answer = answer_question(''.join(transcript_text_list), prompt)
            timestamp = get_answer_location_in_video(answer, transcript_text_list, transcript)
            
            st.session_state.time_stamp = timestamp
            st.session_state.messages.append({"role":"assistant", "content":f"{timestamp}, {answer}"})
            messages.chat_message("assistant").write(f"{timestamp}, {answer}")

            st.rerun()


        else:
            # time.sleep(1)
            st.session_state.messages.append({"role":"assistant", "content":"Please enter a video url!"})
            messages.chat_message("assistant").write("Please enter a video url!")
        


question_answering_model = "deepset/minilm-uncased-squad2"
sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'



# Fetches video transcript given the video url
def download_transcript(video_url):
    video_id = video_url.split('=')[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text_list = [section['text'] for section in transcript]
    return transcript, transcript_text_list

# Return the most likely answer to a prompt in a prticular context
# using the specified QA_model
def answer_question(context, question):
    model = pipeline('question-answering', model=question_answering_model, tokenizer=question_answering_model)
    model_input = {'question': question, 'context': context}
    answer = model(model_input)['answer']
    return answer

# Get the time in the video where the answer was said
def get_answer_location_in_video(answer, transcript_text_list, transcript):
    model = SentenceTransformer(question_answering_model)
    transcript_embeddings = model.encode(transcript_text_list)
    answer_embedding = model.encode(answer)
    similarity = cosine_similarity(transcript_embeddings, answer_embedding.reshape(1,-1))
    best_match_idx = np.argmax(np.squeeze(similarity))
    best_match = transcript_text_list[best_match_idx]
    best_match_start_time = transcript[best_match_idx]['start']

    return best_match_start_time





if __name__ == "__main__":
    main()