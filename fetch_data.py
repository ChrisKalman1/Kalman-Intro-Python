import mediacloud.api as mc
import pandas as pd
from tqdm import tqdm
from config import API_KEY, COLLECTIONS, KEYWORDS, START_DATE, END_DATE, VOLUME_FILE, ARTICLES_FILE

mc_search = mc.SearchApi(API_KEY)

def fetch_volume_over_time():
    """
    Fetches monthly article counts mentioning homelessness
    for each country collection across the full date range.
    Saves results to VOLUME_FILE.
    """
    all_counts = []

    for country, collection_id in COLLECTIONS.items():
        print(f"Fetching volume data for {country}...")

        counts = mc_search.story_count_over_time(
            KEYWORDS,
            START_DATE,
            END_DATE,
            collection_ids=[collection_id]
        )

        for entry in counts:
            all_counts.append({
                "country": country,
                "date": entry["date"],
                "count": entry["count"]
            })

    df = pd.DataFrame(all_counts)
    df.to_csv(VOLUME_FILE, index=False)
    print(f"Volume data saved to {VOLUME_FILE}")
    return df


def fetch_sampled_articles(articles_per_year=150):
    """
    Fetches a stratified sample of articles per country per year.
    Pulls title, URL, publication date, and source for each article.
    Saves results to ARTICLES_FILE.
    """
    import datetime
    all_articles = []

    years = list(range(2014, 2025))

    for country, collection_id in COLLECTIONS.items():
        print(f"\nFetching articles for {country}...")

        for year in tqdm(years, desc=f"{country} years"):
            start = f"{year}-01-01"
            end = f"{year}-12-31"

            try:
                stories = mc_search.story_list(
                    KEYWORDS,
                    start,
                    end,
                    collection_ids=[collection_id],
                    limit=articles_per_year
                )

                for story in stories:
                    all_articles.append({
                        "country": country,
                        "year": year,
                        "title": story.get("title", ""),
                        "url": story.get("url", ""),
                        "publish_date": story.get("publish_date", ""),
                        "source": story.get("media_name", ""),
                        "text": story.get("text", "")
                    })

            except Exception as e:
                print(f"Error fetching {country} {year}: {e}")
                continue

    df = pd.DataFrame(all_articles)
    df.to_csv(ARTICLES_FILE, index=False)
    print(f"\nSampled articles saved to {ARTICLES_FILE}")
    return df


if __name__ == "__main__":
    fetch_volume_over_time()
    fetch_sampled_articles()