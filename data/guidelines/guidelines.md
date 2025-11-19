# Annotation Guidelines

## Introduction

These guidelines help annotators evaluate the relevance of search results for user queries. Use a 3-point scale: 0 (Not Relevant), 1 (Somewhat Relevant), 2 (Highly Relevant).

## General Principles

Always consider user intent when evaluating relevance. The candidate should satisfy what the user is looking for, not just contain matching keywords.

### User Intent Types

- **Navigational**: User wants to find a specific item (artist, album, song)
- **Informational**: User wants to learn about something
- **Transactional**: User wants to perform an action (buy, download, subscribe)

## Music Search Guidelines

### Album Relevance

When the query mentions an album name, the candidate should be that specific album.

#### Studio vs Live Albums

**Section 4.1**: For general album queries without explicit intent signals, studio albums are acceptable.

**Section 4.2**: When the query explicitly mentions "live", "en vivo", "concert", or "tour", the candidate must be a live recording. Studio albums are not acceptable for live-specific queries.

Example:
- Query: "beatles abbey road" → Studio album "Abbey Road" is Highly Relevant (2)
- Query: "beatles live at hollywood bowl" → Live album required, studio albums score 0

### Artist Pages

**Section 5.1**: For queries seeking an artist, the official artist page or profile is Highly Relevant (2).

**Section 5.2**: Individual albums/songs by the artist are Somewhat Relevant (1) unless the query specifically asks for them.

## Locale-Specific Rules

### Spanish (es-ES)

**Section 6.1**: "en vivo" is equivalent to "live" in English and requires live recordings.

**Section 6.2**: Typos and accent variations are common. Be flexible with minor spelling differences.

## Device Context

### Mobile

**Section 7.1**: Mobile users often use shorter queries. Infer intent from context.

### Desktop

**Section 7.2**: Desktop users typically provide more detailed queries.

## Edge Cases

### Compilations and Greatest Hits

**Section 8.1**: For queries seeking specific albums, compilation albums are Somewhat Relevant (1) unless the query explicitly asks for compilations.

### Remasters and Deluxe Editions

**Section 8.2**: Remastered or deluxe editions of an album are considered equivalent to the original for relevance scoring.

