

-- run with `mysql --local_infile=1 -u root -p < read_table.sql`


SET GLOBAL local_infile = 1;

CREATE DATABASE IF NOT EXISTS arxiv_recommender;
USE arxiv_recommender;

DROP TABLE IF EXISTS papers;
CREATE TABLE papers (
    id INT UNSIGNED PRIMARY KEY,
    paper_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT,
    categories TEXT,
    update_year SMALLINT
)
engine=InnoDB, character set utf8mb4, collate utf8mb4_unicode_ci;

DROP TABLE IF EXISTS embeddings;
CREATE TABLE embeddings (
    id INT UNSIGNED PRIMARY KEY,
    text_for_embedding TEXT NOT NULL
)
engine=InnoDB, character set utf8mb4, collate utf8mb4_unicode_ci;



SET FOREIGN_KEY_CHECKS = 0;
SET UNIQUE_CHECKS = 0;

LOAD DATA LOCAL INFILE '/Users/salirafi/Documents/Personal Project/Abstract Recommender/data/arxiv_metadata.csv'
INTO TABLE papers
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"'
ESCAPED BY '\\'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(id, paper_id, title, abstract, authors, categories, update_year);

LOAD DATA LOCAL INFILE '/Users/salirafi/Documents/Personal Project/Abstract Recommender/data/arxiv_embeddings.csv'
INTO TABLE embeddings
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"'
ESCAPED BY '\\'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(id, text_for_embedding);

SET FOREIGN_KEY_CHECKS = 1;
SET UNIQUE_CHECKS = 1;