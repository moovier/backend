# Moovier - Movie Recommendation System

Moovier is a novel movie recommendation system that uses [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) to offer personalized movie suggestions.

### Installation

To use this application, an [TMDB API](https://www.themoviedb.org/) key is required to fetch information regarding the recommended movies.

   ```bash
   git clone https://github.com/moovier/backend.git     # clone back-end
   git clone https://github.com/moovier/frontend.git    # clone front-end
  
   python -m venv moovier                               # create moovier venv
   source moovier/bin/activate                          # activate moovier

   export TMDB_API_KEY=secret-key                       # set-up tmdb key
   cd moovier/backend                                   # navigate to back-end
   pip install -r src/requirements.txt                  # install dependencies
   dvc pull -r myremote                                 # pulls the dataset from Google Drive
   
   cd ../frontend                                       # navigate to front-end
   npm install                                          # install dependencies
   ```

### Usage

The project can be either run as a standalone [Kedro](https://kedro.org/) pipeline using `kedro run --to-outputs "recommended_movies"` or as
a back-end API using `uvicorn app:app`. 
To obtain recommendations based on a model hyper-tuned by [Optuna](https://optuna.org/) run `kedro run --to-outputs "optuna_recommended_movies"`. 
The application exposes the following endpoints:

 - `/models` serves the names of pre-trained models that can be used for inference or fine-tuning.

```bash
   curl -X 'GET' \
   'http://127.0.0.1:8000/models' \
   -H 'accept: application/json'
```

```json
["moovier_emb_50_trained_0", "moovier_emb_25_trained_0", "moovier_emb_10_trained_0"]
```
   
 - `/predict` is used for returning movie recommendations. `model_name` specifies which of the pre-trained models to use for inference, while `top_k` selects how many movies to return for each `user_id` passed in the body.
   
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict?model_name=moovier_emb_25_trained_0&top_k=3' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  389, 456
]'
```

 ```json
{
  "389": [
    {
      "adult": false,
      "backdrop_path": "/8lBcqakI3F19NWkXdqE0M8W76b9.jpg",
      "belongs_to_collection": null,
      "budget": 72000000,
      "genres": [
        {
          "id": 28,
          "name": "Action"
        },
        {
          "id": 18,
          "name": "Drama"
        },
        {
          "id": 36,
          "name": "History"
        },
        {
          "id": 10752,
          "name": "War"
        }
      ],
      "homepage": "",
      "id": 197,
      "imdb_id": "tt0112573",
      "original_language": "en",
      "original_title": "Braveheart",
      "overview": "Enraged at the slaughter of Murron, his new bride and childhood love, Scottish warrior William Wallace slays a platoon of the local English lord's soldiers. This leads the village to revolt and, eventually, the entire country to rise up against English rule.",
      "popularity": 60.035,
      "poster_path": "/or1gBugydmjToAEq7OZY0owwFk.jpg",
      "production_companies": [
        {
          "id": 4564,
          "logo_path": null,
          "name": "Icon Entertainment International",
          "origin_country": "GB"
        },
        {
          "id": 7965,
          "logo_path": null,
          "name": "The Ladd Company",
          "origin_country": "US"
        },
        {
          "id": 11353,
          "logo_path": null,
          "name": "B.H. Finance C.V.",
          "origin_country": ""
        }
      ],
      "production_countries": [
        {
          "iso_3166_1": "US",
          "name": "United States of America"
        }
      ],
      "release_date": "1995-05-24",
      "revenue": 213216216,
      "runtime": 177,
      "spoken_languages": [
        {
          "english_name": "English",
          "iso_639_1": "en",
          "name": "English"
        },
        {
          "english_name": "French",
          "iso_639_1": "fr",
          "name": "Français"
        },
        {
          "english_name": "Latin",
          "iso_639_1": "la",
          "name": "Latin"
        },
        {
          "english_name": "Gaelic",
          "iso_639_1": "gd",
          "name": ""
        }
      ],
      "status": "Released",
      "tagline": "Every man dies. Not every man truly lives.",
      "title": "Braveheart",
      "video": false,
      "vote_average": 7.936,
      "vote_count": 9488
    },
    {
      "adult": false,
      "backdrop_path": "/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg",
      "belongs_to_collection": null,
      "budget": 25000000,
      "genres": [
        {
          "id": 18,
          "name": "Drama"
        },
        {
          "id": 80,
          "name": "Crime"
        }
      ],
      "homepage": "",
      "id": 278,
      "imdb_id": "tt0111161",
      "original_language": "en",
      "original_title": "The Shawshank Redemption",
      "overview": "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope.",
      "popularity": 141.475,
      "poster_path": "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
      "production_companies": [
        {
          "id": 97,
          "logo_path": "/7znWcbDd4PcJzJUlJxYqAlPPykp.png",
          "name": "Castle Rock Entertainment",
          "origin_country": "US"
        }
      ],
      "production_countries": [
        {
          "iso_3166_1": "US",
          "name": "United States of America"
        }
      ],
      "release_date": "1994-09-23",
      "revenue": 28341469,
      "runtime": 142,
      "spoken_languages": [
        {
          "english_name": "English",
          "iso_639_1": "en",
          "name": "English"
        }
      ],
      "status": "Released",
      "tagline": "Fear can hold you prisoner. Hope can set you free.",
      "title": "The Shawshank Redemption",
      "video": false,
      "vote_average": 8.707,
      "vote_count": 25123
    },
    {
      "adult": false,
      "backdrop_path": "/kOPYCJJVodXk76DutR217BkundU.jpg",
      "belongs_to_collection": {
        "id": 609766,
        "name": "Patton Collection",
        "poster_path": "/cX15TXfPzxqaWa4s95FZvMmRFy0.jpg",
        "backdrop_path": null
      },
      "budget": 12000000,
      "genres": [
        {
          "id": 10752,
          "name": "War"
        },
        {
          "id": 18,
          "name": "Drama"
        },
        {
          "id": 36,
          "name": "History"
        }
      ],
      "homepage": "",
      "id": 11202,
      "imdb_id": "tt0066206",
      "original_language": "en",
      "original_title": "Patton",
      "overview": "\"Patton\" tells the tale of General George S. Patton, famous tank commander of World War II. The film begins with patton's career in North Africa and progresses through the invasion of Germany and the fall of the Third Reich. Side plots also speak of Patton's numerous faults such his temper and habit towards insubordination.",
      "popularity": 32.092,
      "poster_path": "/rLM7jIEPTjj4CF7F1IrzzNjLUCu.jpg",
      "production_companies": [
        {
          "id": 25,
          "logo_path": "/qZCc1lty5FzX30aOCVRBLzaVmcp.png",
          "name": "20th Century Fox",
          "origin_country": "US"
        }
      ],
      "production_countries": [
        {
          "iso_3166_1": "US",
          "name": "United States of America"
        }
      ],
      "release_date": "1970-01-25",
      "revenue": 89800000,
      "runtime": 172,
      "spoken_languages": [
        {
          "english_name": "Arabic",
          "iso_639_1": "ar",
          "name": "العربية"
        },
        {
          "english_name": "English",
          "iso_639_1": "en",
          "name": "English"
        },
        {
          "english_name": "French",
          "iso_639_1": "fr",
          "name": "Français"
        },
        {
          "english_name": "German",
          "iso_639_1": "de",
          "name": "Deutsch"
        },
        {
          "english_name": "Italian",
          "iso_639_1": "it",
          "name": "Italiano"
        },
        {
          "english_name": "Russian",
          "iso_639_1": "ru",
          "name": "Pусский"
        }
      ],
      "status": "Released",
      "tagline": "The Rebel Warrior",
      "title": "Patton",
      "video": false,
      "vote_average": 7.527,
      "vote_count": 993
    }
  ],
  "456": [
    {
      "adult": false,
      "backdrop_path": "/3f92DMBTFqr3wgXpfxzrb0qv8nG.jpg",
      "belongs_to_collection": null,
      "budget": 22000000,
      "genres": [
        {
          "id": 18,
          "name": "Drama"
        },
        {
          "id": 36,
          "name": "History"
        },
        {
          "id": 10752,
          "name": "War"
        }
      ],
      "homepage": "http://www.schindlerslist.com/",
      "id": 424,
      "imdb_id": "tt0108052",
      "original_language": "en",
      "original_title": "Schindler's List",
      "overview": "The true story of how businessman Oskar Schindler saved over a thousand Jewish lives from the Nazis while they worked as slaves in his factory during World War II.",
      "popularity": 80.554,
      "poster_path": "/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",
      "production_companies": [
        {
          "id": 56,
          "logo_path": "/cEaxANEisCqeEoRvODv2dO1I0iI.png",
          "name": "Amblin Entertainment",
          "origin_country": "US"
        }
      ],
      "production_countries": [
        {
          "iso_3166_1": "US",
          "name": "United States of America"
        }
      ],
      "release_date": "1993-12-15",
      "revenue": 321365567,
      "runtime": 195,
      "spoken_languages": [
        {
          "english_name": "German",
          "iso_639_1": "de",
          "name": "Deutsch"
        },
        {
          "english_name": "Polish",
          "iso_639_1": "pl",
          "name": "Polski"
        },
        {
          "english_name": "Hebrew",
          "iso_639_1": "he",
          "name": "עִבְרִית"
        },
        {
          "english_name": "English",
          "iso_639_1": "en",
          "name": "English"
        }
      ],
      "status": "Released",
      "tagline": "Whoever saves one life, saves the world entire.",
      "title": "Schindler's List",
      "video": false,
      "vote_average": 8.573,
      "vote_count": 14891
    },
    {
      "adult": false,
      "backdrop_path": "/mx2jS5Kaa5rmaldEFzKeKpDN9Q2.jpg",
      "belongs_to_collection": null,
      "budget": 15000000,
      "genres": [
        {
          "id": 12,
          "name": "Adventure"
        },
        {
          "id": 18,
          "name": "Drama"
        },
        {
          "id": 36,
          "name": "History"
        },
        {
          "id": 10752,
          "name": "War"
        }
      ],
      "homepage": "",
      "id": 947,
      "imdb_id": "tt0056172",
      "original_language": "en",
      "original_title": "Lawrence of Arabia",
      "overview": "The story of British officer T.E. Lawrence's mission to aid the Arab tribes in their revolt against the Ottoman Empire during the First World War. Lawrence becomes a flamboyant, messianic figure in the cause of Arab unity but his psychological instability threatens to undermine his achievements.",
      "popularity": 35.558,
      "poster_path": "/AiAm0EtDvyGqNpVoieRw4u65vD1.jpg",
      "production_companies": [
        {
          "id": 11356,
          "logo_path": null,
          "name": "Horizon Pictures",
          "origin_country": "GB"
        }
      ],
      "production_countries": [
        {
          "iso_3166_1": "GB",
          "name": "United Kingdom"
        }
      ],
      "release_date": "1962-12-11",
      "revenue": 69995385,
      "runtime": 228,
      "spoken_languages": [
        {
          "english_name": "English",
          "iso_639_1": "en",
          "name": "English"
        },
        {
          "english_name": "Arabic",
          "iso_639_1": "ar",
          "name": "العربية"
        },
        {
          "english_name": "Turkish",
          "iso_639_1": "tr",
          "name": "Türkçe"
        }
      ],
      "status": "Released",
      "tagline": "A mighty motion picture of action and adventure!",
      "title": "Lawrence of Arabia",
      "video": false,
      "vote_average": 7.985,
      "vote_count": 2734
    },
    {
      "adult": false,
      "backdrop_path": "/7dX07NQv4p6kfzSkA9rf9LvlvVD.jpg",
      "belongs_to_collection": null,
      "budget": 2200000,
      "genres": [
        {
          "id": 53,
          "name": "Thriller"
        },
        {
          "id": 18,
          "name": "Drama"
        }
      ],
      "homepage": "",
      "id": 982,
      "imdb_id": "tt0056218",
      "original_language": "en",
      "original_title": "The Manchurian Candidate",
      "overview": "Near the end of the Korean War, a platoon of U.S. soldiers is captured by communists and brainwashed. Following the war, the platoon is returned home, and Sergeant Raymond Shaw is lauded as a hero by the rest of his platoon. However, the platoon commander, Captain Bennett Marco, finds himself plagued by strange nightmares and soon races to uncover a terrible plot.",
      "popularity": 25.423,
      "poster_path": "/h0mkK00GjBCJoBYm3yvPdkzlIyV.jpg",
      "production_companies": [
        {
          "id": 623,
          "logo_path": null,
          "name": "MC Productions",
          "origin_country": ""
        },
        {
          "id": 60,
          "logo_path": "/1SEj4nyG3JPBSKBbFhtdcHRaIF9.png",
          "name": "United Artists",
          "origin_country": "US"
        }
      ],
      "production_countries": [
        {
          "iso_3166_1": "US",
          "name": "United States of America"
        }
      ],
      "release_date": "1962-10-24",
      "revenue": 7700000,
      "runtime": 126,
      "spoken_languages": [
        {
          "english_name": "English",
          "iso_639_1": "en",
          "name": "English"
        }
      ],
      "status": "Released",
      "tagline": "When you've seen it all, you'll swear there's never been anything like it!",
      "title": "The Manchurian Candidate",
      "video": false,
      "vote_average": 7.55,
      "vote_count": 633
    }
  ]
}
```

 - `/train` fine-tunes the selected `model_name` with a new batch of data. `validation_split` can be used to create a validation dataset for evaluation purposes; `patience` specifies at which point to activate the early stopping callback. By default, the `val_loss` is being monitored. The model is fine-tuned on top of the body request.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/train?model_name=moovier_emb_10_trained_0&validation_split=0.2&patience=10' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "users": [
    498, 389, 456, 203, 446, 252, 214, 268, 115, 333, 594, 395, 218, 436, 520, 301, 130, 387, 174, 596
  ],
  "movies": [
    5268, 3699, 4617, 2702, 2437, 1298, 241, 569, 3787, 7103, 2521, 6478, 1988, 2651, 421, 6597, 6732, 1984, 1583, 8265
  ],
  "ratings": [
    1, 4, 5, 3, 2, 2, 3, 4, 3, 5, 1, 3, 5, 5, 4, 4, 1, 2, 4, 5
  ]
}'
```

```json
"moovier_emb_10_trained_1.h5"
```

