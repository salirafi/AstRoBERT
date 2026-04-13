SET NAMES utf8mb4;

DROP TABLE IF EXISTS papers;
CREATE TABLE papers (
    id BIGINT NOT NULL,
    paper_id VARCHAR(32) NOT NULL,
    title MEDIUMTEXT NOT NULL,
    authors JSON NOT NULL,
    categories JSON NOT NULL,
    update_year SMALLINT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_papers_paper_id (paper_id),
    KEY idx_papers_update_year (update_year)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

--
--   python generate_mysql_inserts.py
--
--   mysql --host=... --user=... --password=... --database=... < load_arxiv_metadata.sql
--   mysql --host=... --user=... --password=... --database=... < inserts_arxiv_metadata.sql
