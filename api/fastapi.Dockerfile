FROM ubuntu

RUN apt update && apt install python3 python3-pip libpq-dev -y
RUN python3 -m pip install fastapi psycopg2-binary --break-system-packages
RUN python3 -m pip install fastapi[standard] --break-system-packages

COPY backend/ /app

EXPOSE 80

WORKDIR /app

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]