#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Mar 29, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : adaily.py
project  : auto_paper
license  : GPL-3.0+

This is the script to download the daily paper.
"""
# Standard Library
import datetime as dt
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

# Types
from typing import Literal, cast, TypedDict

# Utils
from rich import print

# Database
import shelve

# Others
import arxiv, httpx, tyro


class Src(Enum):
    """Source."""

    arxiv = "arxiv"
    biorxiv = "biorxiv"
    medrxiv = "medrxiv"


@dataclass
class Conf:
    """Configuration."""

    src: Src = Src.arxiv
    max_paper: int = 100
    paper_db: Path = Path("./papers.db")
    pdf_dir: Path = Path("./pdfs/")
    query: str = "cat:cs.AI"
    filters: Literal["all", "new", "today"] = "new"
    sortby: arxiv.SortCriterion = arxiv.SortCriterion.LastUpdatedDate
    filterby: Literal["updated", "published"] = "updated"
    save_pdf: bool = True
    show: bool = True


class Paper(TypedDict):
    """Paper."""

    pid: str
    title: str
    abstract: str
    published: str
    updated: str
    categorie: tuple[str, ...]


TODAY = dt.datetime.now(tz=dt.UTC).date()


def paperdb(db: Path) -> dict[str, Paper]:
    """Get the paper database."""
    if not db.exists():
        db.parent.mkdir(parents=True, exist_ok=True)

    with shelve.open(db.as_posix()) as pdb:  # noqa: S301
        return dict(pdb)


def savedb(papers: tuple[Paper, ...], db: Path) -> None:
    """Save the paper database."""
    with shelve.open(db.as_posix()) as pdb:  # noqa: S301
        pdb.update(dict(map(lambda p: (p["pid"], p), papers)))
        pdb.sync()


def filter_papers(paper: arxiv.Result, db: dict[str, Paper], conf: Conf) -> bool:
    """Filter the papers."""
    match conf.filters:
        case "all":
            return True
        case "new":
            return paper.entry_id not in db
        case "today":
            return (
                vars(paper).get(conf.filterby, dt.datetime.now(dt.UTC)).date() == TODAY
            )


def get_papers(conf: Conf) -> tuple[Paper, ...]:
    """Get all the papers."""
    db = paperdb(conf.paper_db)
    results = arxiv.Search(
        query=conf.query,
        max_results=conf.max_paper,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
    ).results()
    ff = partial(filter_papers, db=db, conf=conf)

    def _get_paper(paper: arxiv.Result) -> Paper:
        return Paper(
            pid=paper.entry_id,
            title=paper.title,
            abstract=paper.summary,
            published=paper.published.strftime("%Y-%m-%d"),
            updated=paper.updated.strftime("%Y-%m-%d"),
            categorie=tuple(paper.categories),
        )

    return tuple(map(_get_paper, filter(ff, results)))


def bmrxiv_query(server: Literal[Src.biorxiv, Src.medrxiv]) -> list[dict[str, str]]:
    """Get the query for biorxiv and medrxiv."""
    delta = dt.timedelta(days=365)
    start = (TODAY - delta).strftime("%Y-%m-%d")
    end = TODAY.strftime("%Y-%m-%d")
    url = f"https://api.biorxiv.org/pubs/{server.value}/{start}/{end}"
    return httpx.get(url).json()["collection"]


def filter_bmpaper(paper: dict[str, str], db: dict[str, Paper], conf: Conf) -> bool:
    """Filter the papers."""
    url = "https://www.biorxiv.org/content/"
    match conf.filters:
        case "all":
            return True
        case "new":
            return (url + paper["preprint_doi"]) not in db
        case "today":
            return (
                dt.datetime.strptime(
                    paper[
                        {"updated": "preprint_date", "published": "published_date"}[
                            conf.filterby
                        ]
                    ],
                    "%Y-%m-%d",
                )
                .replace(tzinfo=dt.UTC)
                .date()
                == TODAY
            )


def get_papers_bmrxiv(conf: Conf) -> tuple[Paper, ...]:
    """Get all the papers."""
    src = cast(Literal[Src.biorxiv, Src.medrxiv], conf.src)
    db = paperdb(conf.paper_db)
    res = bmrxiv_query(src)[: conf.max_paper]
    ff = partial(filter_bmpaper, db=db, conf=conf)
    url = f"https://www.{conf.src.value}.org/content/"

    def _get_paper(paper: dict[str, str]) -> Paper:
        return Paper(
            pid=url + paper["preprint_doi"],
            title=paper["preprint_title"],
            abstract=paper["preprint_abstract"],
            published=paper["published_date"],
            updated=paper["preprint_date"],
            categorie=(paper["preprint_category"],),
        )

    return tuple(map(_get_paper, filter(ff, res)))


def pdf(url: str, conf: Conf, src: Src | Literal["free"]) -> None:
    """Download the pdf."""
    match src:
        case Src.arxiv:
            name = url.split("/")[-1]
            if not url.endswith("pdf"):
                url = f"{url}.pdf"
        case Src.biorxiv:
            name = ".".join(url.split("/")[-2:])
            if not url.endswith("pdf"):
                url = f"{url}.full.pdf"
        case Src.medrxiv:
            name = ".".join(url.split("/")[-2:])
            if not url.endswith("pdf"):
                url = f"{url}.full.pdf"
        case "free":
            name = url.split("/")[-1]
        case _:
            raise ValueError(f"Unknown source: {src}")
    pdf = httpx.get(url, follow_redirects=True, timeout=60)
    conf.pdf_dir.mkdir(parents=True, exist_ok=True)
    if not name.endswith("pdf"):
        name += ".pdf"
    with open(conf.pdf_dir / name, "wb") as f:
        f.write(pdf.content)


def paper(conf: Conf) -> tuple[Paper, ...]:
    """Run main function."""
    match conf.src:
        case Src.arxiv:
            ps = get_papers(conf)
        case Src.biorxiv:
            ps = get_papers_bmrxiv(conf)
        case Src.medrxiv:
            ps = get_papers_bmrxiv(conf)
        case _:
            raise ValueError(f"Unknown source: {conf.src}")
    if conf.save_pdf:
        for p in ps:
            if conf.src == Src.arxiv:
                url = p["pid"].replace("/abs/", "/pdf/")
            else:
                url = f"{p['pid']}.full.pdf"
            if conf.show:
                print(f"Fetching {url}")
            pdf(url, conf, conf.src)
    savedb(ps, conf.paper_db)
    return ps


def main(conf: Conf, url: str | None = None) -> None:
    """Run main function."""
    if url is not None:
        if "arxiv" in url:
            pdf(url.replace("/abs/", "/pdf/"), conf, Src.arxiv)
        elif "biorxiv" in url:
            pdf(url, conf, Src.biorxiv)
        elif "medrxiv" in url:
            pdf(url, conf, Src.medrxiv)
        else:
            pdf(url, conf, "free")
        return
    paper(conf)


if __name__ == "__main__":
    tyro.cli(main)
