import argparse
from langchain_community.llms.ollama import Ollama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    model = Ollama(model="llama3")
    response_text = model.invoke(query_text)
    print(response_text)


if __name__ == "__main__":
    main()
