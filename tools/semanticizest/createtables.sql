pragma foreign_keys = on;
pragma journal_mode = off;
pragma synchronous = off;

drop table if exists linkstats;
drop table if exists ngrams;

create table parameters (
    key text primary key not NULL,
    value text default NULL
);

create table ngrams (
    id integer primary key default NULL,
    ngram text unique not NULL,
    tf integer default 0,
    df integer default 0
);

create table linkstats (
    ngram_id integer not NULL,
    target text not NULL,
    count integer not NULL,
    foreign key(ngram_id) references ngrams(id)
);

create index link_target on linkstats(target);
