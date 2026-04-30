CREATE EXTENSION vector;

CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title TEXT,
    description TEXT,
    embedding vector(384)
);

ALTER TABLE books ADD CONSTRAINT unique_title UNIQUE (title);
