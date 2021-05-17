# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import json
import requests

from imdb import IMDb
import wikipedia
from kvtoraman_scraper.proxy_kon import PROXY

from config import TMDB_KEY, OMDB_KEY

ia = IMDb('http', proxy=PROXY)


def search_movie(movie_title, year=None):
    # candidates = ia.search_movie(movie_title)
    # return [IMDBMovieMetaData(candidate.movieID) for candidate in candidates]

    query = f'https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}' \
            f'&language=en-US&query={movie_title}&page=1&include_adult=false'
    if year:
        query += f'&year={year}'
    r = requests.get(query)
    res = json.loads(r.content, encoding='utf-8')
    return [OpenMovieMetaData(item['id']) for item in res['results']]


def get_movie_by_id(imdb_id):
    r = requests.get(f'http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_KEY}')
    res = json.loads(r.content, encoding='utf-8')
    for candidate in search_movie(res['Title'], res['Year']):
        if candidate.imdb_id == imdb_id:
            return candidate
    raise ValueError(f"Cannot find the movie by id: {imdb_id}")


class OpenMovieMetaData:
    def __init__(self, movie_id):
        r = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_KEY}&language=en-US')
        res = json.loads(r.content, encoding='utf-8')
        self.title = res['title']
        self.imdb_id = res['imdb_id']
        self.year = res["release_date"].split("-")[0]

        r = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_KEY}&language=en-US')
        res = json.loads(r.content, encoding='utf-8')
        self.cast = set()
        self.cast_clean = []
        for item in res['cast']:
            self.cast |= set(sum([role.data['name'].upper().split() for role in item['character']], []))
            self.cast_clean.append(item['character'])

    def get_plot(self):
        # try to get plot from wikipedia
        try:
            page = wikipedia.page(f"{self.title} ({self.year} film)")
            if not page.title.startswith(self.title):
                wiki_plot = ''
            else:
                start = page.content.index("== Plot ==") + 10  # Plot starts
                stop = page.content.index("== ", start)
                wiki_plot = page.content[start: stop].splitlines()
                wiki_plot = " ".join(wiki_plot)
        except wikipedia.PageError:
            wiki_plot = ''

        # try to get plot from OMDB
        r = requests.get(f'http://www.omdbapi.com/?i={self.imdb_id}&apikey={OMDB_KEY}&plot=full')
        res = json.loads(r.content, encoding='utf-8')
        omdb_plot = res['Plot'] if 'Plot' in res else ""

        if len(wiki_plot) > len(omdb_plot):
            return wiki_plot
        else:
            return omdb_plot


class IMDBMovieMetaData:
    """
    We are strongly discourage usage of IMDB due to their data collection policy.
    """
    def __init__(self, imdb_id):
        self.imdb_id = imdb_id
        movie = ia.get_movie(imdb_id)
        self.title = movie.data['title']
        self.cast = set()
        self.cast_clean = []

        from imdb.Character import Character
        if 'cast' in movie.data:
            for actor in movie.data['cast']:
                current_role = actor.currentRole
                if isinstance(current_role, Character):
                    current_role = [current_role]
                self.cast |= set(sum([role.data['name'].upper().split() for role in current_role
                                      if 'name' in role.data], []))
                clean_role = ' '.join(role.data['name'] for role in current_role if 'name' in role.data)
                if clean_role:
                    self.cast_clean.append(clean_role)

    def get_plot(self):
        movie = ia.get_movie(self.imdb_id)
        plot = movie.get('synopsis')
        if plot:
            return plot[0]
        else:
            plot = movie.get('plot summary')
            return plot[0] if plot else ''

