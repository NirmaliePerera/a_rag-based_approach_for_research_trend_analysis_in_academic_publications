# RAG-Based Topic Trend Analysis in Academic Publications

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/NirmaliePerera/rag-based_topic_trend_analysis_in_academic_publications)

## Overview

This project presents a Retrieval-Augmented Generation (RAG) based approach for analyzing topic trends within a user-provided corpus of academic publications. The prototype supports creating a knowledge base from research papers provided as input, analyzing topic trends based on the content of paper abstracts, analyzing the evolution of topics over time, and presenting identified trends through summarization and visualization to support easier interpretation and decision-making.

The core architecture underlying this system is Retrieval-Augmented Generation (RAG), a technique that integrates two distinct processes: the retrieval of relevant information from an external knowledge source, and the generation of natural-language output by an LLM conditioned on that retrieved information.

The approach combines Natural Language Processing (NLP), semantic embeddings, vector databases, topic modeling, and Large Language Models (LLMs) to support research trend analysis and topic evolution analysis.

The project is developed as an undergraduate research project for the BSc (Hons) in Information Technology degree.

## Research Problem

The increasing volume of academic publications makes it difficult for researchers to manually identify emerging research topics and understand how research areas change over time.

Traditional keyword-based approaches may not adequately capture the semantic relationships between research topics. This project investigates how a RAG-based architecture combined with NLP and topic modeling techniques can be used to identify and analyze research trends from academic publications.

## Research Objectives

* To collect and preprocess academic publications and extract relevant metadata such as title, abstract, and keywords using text mining techniques.
* To design a Retrieval-Augmented Generation (RAG)-based framework for automated research trend analysis.
* To develop an NLP-based method for keyword and topic tracking to identify emerging research trends within academic publications.
* To implement a prototype system that integrates metadata extraction, RAG-based analysis, and trend visualization.
* To evaluate the effectiveness of the proposed model in identifying research trends using appropriate evaluation metrics.

## System Architecture

The system is composed of two main components: knowledge base creation and RAG-based trend analysis.

In knowledge base creation, uploaded research paper PDFs undergo metadata extraction (via the Gemini API), preprocessing, and embedding generation (using a lightweight Sentence Transformer model), with embeddings stored in a vector database. BERTopic is applied to discover underlying research themes across the corpus.

In RAG-based trend analysis, a user-provided keyword or phrase is used to retrieve semantically related papers via similarity search against stored embeddings. Retrieved content is then summarized using an LLM, and visualized through interactive charts alongside a table of related paper details.

A separate evaluation component tracks how discovered topics evolve year-over-year, classifying them as persisted, grown, declined, or newly emerged.

## Technology Stack

* Language: Python
* LLM: Gemini API (metadata extraction, trend summarization)
* Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
* Topic Modeling: BERTopic (UMAP, HDBSCAN, c-TF-IDF)
* Vector Database: ChromaDB
* Relational Database: SQLite
* Interface: Streamlit
* Visualization: Plotly

## Features

* Automated metadata extraction from research paper PDFs
* Semantic (embedding-based) retrieval of related papers
* Topic modeling across the full corpus and on a per-year basis
* Year-to-year topic evolution analysis (persisted / grown / declined / emerged)
* LLM-generated trend summaries grounded in retrieved paper content
* Interactive visualizations of publication trends and topic distribution
