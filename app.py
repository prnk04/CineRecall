import json
import time
import streamlit as st
from src.rag.user_query_flow import UserQuery
import asyncio

RATE_LIMIT = 60


def start_find():
    st.session_state.disabled = True


def run_find_if_needed():
    if not st.session_state.get("disabled"):
        return

    with st.spinner("Seeking the movie"):
        validate_input()

    st.session_state.disabled = False
    st.rerun()  # safe rerun: no layout emitted yet


def validate_input():

    # Check time difference between requests
    current_time = time.time()
    time_since_last_request = current_time - st.session_state.last_request_time

    user_in = st.session_state.user_input.strip()

    if len(user_in.split()) <= 4:
        st.session_state.invalid_movie_plot = True
        st.session_state.movie_response = {}
        return

    st.session_state.invalid_movie_plot = False
    if time_since_last_request < RATE_LIMIT:
        st.session_state.coolDownPeriod = RATE_LIMIT - time_since_last_request
        st.session_state.rateLimitExceeded = True
        return None
    else:
        st.session_state.coolDownPeriod = 0
        st.session_state.rateLimitExceeded = False
        st.session_state.last_request_time = current_time

    try:
        # movie_response = json.dumps(
        #     {
        #         "movie_title": "Swelter (2014)",
        #         "confidence_score": 0.9,
        #         "explanation": "Why this matches your description: \n1. The plot revolves around a robbery where five masked robbers steal money from a Las Vegas casino. \n2. One of the robbers suffers from amnesia and is tracked down by the rest of the gang, which aligns with the user's mention of a robbery and a user with a fuzzy memory.",
        #     }
        # )
        uq = st.session_state.query_handler
        movie_response = uq.retrieve_movie(user_in)
        movie_res_json = json.loads(movie_response)

        if movie_response is not None:
            movie_res_json = json.loads(movie_response)
            if movie_res_json.get("confidence_score", 0.0) == 0.0:
                st.session_state.movie_response = {
                    "movie_title": "Sorry, we could not find the movie",
                    "explanation": "Seems like I do not have data about the movie yet. Please try some other movie, or add more plot information",
                }
            else:
                st.session_state.movie_response = json.loads(movie_response)

            print("I received: ", st.session_state.movie_response)
        else:
            st.session_state.movie_response = {
                "movie_title": "Sorry, we could not find the movie",
                "explanation": "Seems like I do not have data about the movie yet. Please try some other movie, or add more plot information",
            }

    except Exception as e:
        print("Error while trying to fetch movie ", e)


def create_main_app():

    if "query_handler" not in st.session_state:
        st.session_state.query_handler = UserQuery()

    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    # ---- state init ----
    st.session_state.setdefault("user_input", "")
    st.session_state.setdefault("invalid_movie_plot", False)
    st.session_state.setdefault("movie_response", {})
    st.session_state.setdefault("disabled", False)
    st.session_state.setdefault("rateLimitExceeded", False)
    st.session_state.setdefault("coolDownPeriod", 0)

    # # ---- execution phase FIRST ----
    # run_find_if_needed()

    # ---- layout phase ----
    with st.container():

        st.text_area(
            label="movie_plot",
            label_visibility="hidden",
            key="user_input",
            placeholder="Describe movie here...",
        )

        error_slot = st.empty()

        st.button(
            "Find",
            type="primary",
            disabled=st.session_state.disabled or st.session_state.rateLimitExceeded,
            on_click=start_find,
        )
        error_slot_1 = st.empty()

        # ---- execution phase FIRST ----
        run_find_if_needed()

        if st.session_state.rateLimitExceeded:
            with st.spinner("Rate Limit Exceeded. Please wait "):

                error_slot_1.warning(
                    f"""Please Note 👋\n
                    CineRecall is a personal portfolio project, and each query makes a real LLM API call.
                    To keep costs manageable, I've limited queries to one every 1 minute.\n
                    Thanks for your patience, and I hope you enjoy trying it out!"""
                )

                time.sleep(st.session_state.coolDownPeriod + 2)

                error_slot_1 = st.empty()
                st.session_state.rateLimitExceeded = False
                st.session_state.disabled = False
                st.rerun()

        # ---- feedback ----
        if st.session_state.invalid_movie_plot:
            error_slot.error("Enter a valid movie plot")

        if (
            st.session_state.movie_response
            and st.session_state.get("disabled") == False
        ):
            st.markdown(f"#### 🎬 {st.session_state.movie_response['movie_title']}")

            explanation = st.session_state.movie_response["explanation"].split("\n")
            st.markdown(f"_{explanation[0].strip()}_")

            for line in explanation[1:]:
                st.write(line)


def create_header():
    with st.container(border=False, height="stretch"):
        st.title(":clapper: CineRecall")
        st.markdown(f"*Remember the plot? We'll find the movie*")


def create_footer():
    with st.container(border=False, height="stretch"):
        st.divider()
        st.markdown(
            """
            <div style='text-align: center; color: gray; '>
                Built with ❤️ using LangChain and  OpenAI GPT-4o-mini | 
                <a href='https://github.com/prnk04/CineRecall' target='_blank'>View on GitHub</a>
                
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="CineRecall",
        page_icon=":clapper:",
        layout="wide",
    )

    create_header()
    create_main_app()
    create_footer()


if __name__ == "__main__":
    main()
