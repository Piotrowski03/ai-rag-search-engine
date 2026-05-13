# Rag based Book recommendation system
Implemented the RAG pipeline and integrated a PostgreSQL database acting as a vector database. Conducted a full data processing
workflow, including loading and cleaning CSV data, generating embeddings using language models, and storing them in the
database. Designed a modular system architecture covering vector search, prompt construction, and integration with the Gemini
API language model. Containerized the application using Docker

## Basic requirements
Firstly sign up and create your free gemini api key from
[https://ai.google.dev/](https://ai.google.dev/)

Then create your own cloud base database for example in 
[https://aiven.io/](https://aiven.io/)

--
## 📁 `.env` file
In the main folder create the .env file

```env
PG_DB_PASSWORD=Your database password
PG_DB_USER=Username for your database
PG_DB_DATABASE=Name of the database
PG_DB_HOST=Host
PG_DB_PORT=Port
GEMINI_API_KEY=Your uniquely generated Gemini api key
```

## Download Docker
Download Docker from
[https://www.docker.com/](https://www.docker.com/)


## In bash
In bash type command
```
docker build -t moja-apka .

```
after creating your docker image you can create docker container typing
```
docker run -it moja-apka
```

## Status
Project initialized.
