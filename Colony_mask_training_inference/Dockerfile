FROM ubuntu:24.04
WORKDIR /usr/src/acm
COPY . .
RUN apt update
RUN apt -y install git
RUN apt -y install curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv sync
ENTRYPOINT ["/usr/src/acm/run_pipeline.sh"]