import gradio as gr
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure API settings
BACKEND_HOST = os.getenv("BACKEND_HOST")
BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")
API_BASE_URL = os.getenv("BACKEND_API_URL")
API_KEY = os.getenv("OPENAI_API_KEY", None)
ROUTE_OPTIONS = ["auto", "chat", "math_qa", "medical_qa"]

if API_BASE_URL is None and BACKEND_HOST:
    API_BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/v1"

if API_BASE_URL is None or API_KEY is None:
    raise ValueError("API_BASE_URL and API_KEY must be set in the environment variables.")

def post_json(path, payload):
    url = f"{API_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 422:
            return {"error": "Backend validation error. Please check your input format."}

        if response.status_code != 200:
            return {"error": f"Error {response.status_code}: {response.text[:200]}"}

        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error: Could not connect to the backend service. Please check if the backend is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The backend service may be overloaded."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Error: {str(e)}"}


def format_result(result):
    if "error" in result:
        return result["error"], {}

    metadata = {
        key: result.get(key)
        for key in ["route", "provider", "model", "classifier_model", "reason"]
        if result.get(key) is not None
    }
    return result.get("content", "No answer received"), metadata


def route_chat(prompt, route):
    if not prompt.strip():
        return "Please enter a prompt.", {}

    payload = {"messages": [{"role": "user", "content": prompt}]}
    if route != "auto":
        payload["route"] = route

    return format_result(post_json("/chat", payload))


def answer_math_question(question):
    if not question.strip():
        return "Please enter a math question.", {}

    return format_result(post_json("/math-qa", {"question": question}))

def answer_medical_question(question, context):
    if not question.strip():
        return "Please enter a medical question.", {}

    payload = {"question": question}
    if context.strip():
        payload["context"] = context

    return format_result(post_json("/medical-qa", payload))


with gr.Blocks() as routed_chat_tab:
    gr.Markdown("# Routed Chat")
    gr.Markdown("Send a prompt to the FastAPI gateway and let the classifier intelligently route to: TensorRT-LLM chat, vLLM math QA, or vLLM medical QA.")

    route_input = gr.Dropdown(ROUTE_OPTIONS, value="auto", label="Route Override")
    prompt_input = gr.Textbox(label="Prompt", placeholder="Ask a math or medical question...", lines=5)
    route_button = gr.Button("Send Request")
    routed_output = gr.Textbox(label="Answer", lines=6)
    routed_metadata = gr.JSON(label="Routing Metadata")

    route_button.click(
        fn=route_chat,
        inputs=[prompt_input, route_input],
        outputs=[routed_output, routed_metadata],
    )

    gr.Examples(
        [
            ["Solve 12 * 18 and explain briefly.", "auto"],
            ["A patient with hyperthyroidism presents with weight loss and tremor. What is the likely diagnosis?", "auto"],
            ["Integrate x^2 + 3x.", "math_qa"],
            ["Hello, how can you help me today?", "chat"],
            ["What are the common symptoms of diabetes?", "medical_qa"],
        ],
        inputs=[prompt_input, route_input],
        outputs=[routed_output, routed_metadata],
        fn=route_chat,
    )


with gr.Blocks() as math_qa_tab:
    gr.Markdown("# Math QA")
    gr.Markdown("This tab pins requests to the vLLM math route (powered by LoRA adapter).")

    math_input = gr.Textbox(label="Math Question", placeholder="Enter a math question...", lines=4)
    math_button = gr.Button("Solve Math Question")
    math_output = gr.Textbox(label="Answer", lines=6)
    math_metadata = gr.JSON(label="Serving Metadata")

    math_button.click(
        fn=answer_math_question,
        inputs=[math_input],
        outputs=[math_output, math_metadata],
    )

    gr.Examples(
        [
            ["What is 144 divided by 12?"],
            ["Differentiate 3x^2 + 2x - 7."],
        ],
        inputs=[math_input],
        outputs=[math_output, math_metadata],
        fn=answer_math_question,
    )

with gr.Blocks() as medical_qa_tab:
    gr.Markdown("# Medical Question Answering")
    gr.Markdown("Ask a free-form Vietnamese medical question. You can optionally provide extra clinical context.")
    
    question_input = gr.Textbox(
        label="Question",
        placeholder="Enter your medical question here...",
        lines=3
    )
    context_input = gr.Textbox(
        label="Optional Context",
        placeholder="Add symptoms, patient history, lab values, or source context if available...",
        lines=4
    )
    
    answer_button = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Answer", lines=4)
    answer_metadata = gr.JSON(label="Serving Metadata")
    
    answer_button.click(
        fn=answer_medical_question,
        inputs=[question_input, context_input],
        outputs=[answer_output, answer_metadata]
    )
    
    gr.Examples(
        [
            ["Viêm phổi do vi khuẩn thường có những triệu chứng nào?", "Bệnh nhân sốt 39 độ C, ho đàm vàng, đau ngực kiểu màng phổi trong 2 ngày."],
            ["Khi nào nên nghi ngờ sốt xuất huyết Dengue ở người lớn?", "Bệnh nhân sống ở vùng dịch, sốt cao liên tục, đau đầu, đau mỏi cơ khớp và có xuất huyết dưới da nhẹ."],
        ],
        inputs=[question_input, context_input],
        outputs=[answer_output, answer_metadata],
        fn=answer_medical_question
    )

demo = gr.TabbedInterface(
    [routed_chat_tab, math_qa_tab, medical_qa_tab],
    ["Routed Chat", "Math QA", "Medical QA"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("FRONTEND_PORT", "7860")))
