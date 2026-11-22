"""
02. Web evidence retrieval with Tavily.

Queries Tavily with the raw claim text and stores, for every claim, the
formatted search results (title, URL, content snippet and score).

Training pool (run between notebooks 01 and 03), basic search depth:

    python data_preparation/02_web_search_tavily.py \
        --input data/training/sample.jsonl \
        --output data/training/sample_tavily.jsonl \
        --claim_column claim --search_depth basic

Evaluation sets, advanced search depth with a publication cut-off date:

    python data_preparation/02_web_search_tavily.py \
        --input data/raw/test/healthFC_annotated.csv \
        --output data/healthfc/healthFC_annotated_tavily_final.csv \
        --claim_column en_claim --search_depth advanced --end_date 2022-12-31

Columns added to the output:
    evidence_tavily_20_results  all (up to 20) results, formatted
    evidence_tavily_5_results   the first 5 results, formatted
    tavily_results_json         raw Tavily response

Requires TAVILY_API_KEY in the environment.
"""

import argparse
import asyncio
import json
import os

import backoff
import pandas as pd
from tavily import TavilyClient


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="Input file (.jsonl, .json or .csv)")
    parser.add_argument("--output", required=True, help="Output file (.jsonl or .csv)")
    parser.add_argument("--claim_column", default="claim", help="Column with the claim text")
    parser.add_argument("--search_depth", choices=["basic", "advanced"], default="basic")
    parser.add_argument("--end_date", default=None, help="Only return results published up to this date (YYYY-MM-DD)")
    parser.add_argument("--max_results", type=int, default=20)
    parser.add_argument("--max_concurrent", type=int, default=8)
    return parser.parse_args()


def read_table(path):
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path)


def write_table(df, path):
    if path.endswith(".jsonl"):
        df.to_json(path, orient="records", lines=True)
    else:
        df.to_csv(path, index=False)


def build_search_kwargs(args):
    kwargs = dict(
        search_depth=args.search_depth,
        max_results=args.max_results,
        include_raw_content="markdown",
        chunks_per_source=5,
    )
    if args.end_date:
        kwargs["end_date"] = args.end_date
    return kwargs


@backoff.on_exception(
    backoff.expo,
    Exception,  # Catch all exceptions including rate limits
    max_tries=5,
    max_time=300,  # Max 5 minutes of retries
    base=2,
    factor=2,
)
async def fetch_tavily_with_backoff(client, claim, search_kwargs):
    """Fetch search results from Tavily API with backoff"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: client.search(query=claim, **search_kwargs)
    )


async def process_claims_parallel(df, client, claim_column, search_kwargs, max_concurrent=3):
    """Simple parallel processing of claims using Tavily with backoff"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one_claim(claim, index):
        async with semaphore:
            try:
                print(f"Processing claim {index}: {claim[:50]}...")
                result = await fetch_tavily_with_backoff(client, claim, search_kwargs)
                # Add small delay to be gentle with API
                await asyncio.sleep(1)
                return result
            except Exception as e:
                print(f"Failed claim {index} after retries: {str(e)}")
                return {"query": claim, "results": [], "error": str(e)}

    tasks = [
        process_one_claim(row[claim_column], idx)
        for idx, (_, row) in enumerate(df.iterrows())
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for result in results:
        if isinstance(result, Exception):
            final_results.append({"query": "", "results": [], "error": str(result)})
        else:
            final_results.append(result)
    return final_results


def format_tavily_results(tavily_response):
    """Format Tavily API response into a readable string format"""
    if not tavily_response or 'results' not in tavily_response:
        return ""

    formatted_results = []
    for result in tavily_response['results']:
        formatted_result = f"Title: {result.get('title', 'N/A')}\n"
        formatted_result += f"URL: {result.get('url', 'N/A')}\n"
        formatted_result += f"Content: {result.get('content', 'N/A')}\n"
        formatted_result += f"Score: {result.get('score', 'N/A')}\n"
        formatted_results.append(formatted_result)

    return '\n\n'.join(formatted_results)


def extract_first_n_results_tavily(tavily_response, n=5):
    """Extract first N results from Tavily response"""
    if not tavily_response or 'results' not in tavily_response:
        return ""

    limited_results = tavily_response['results'][:n]
    limited_response = {
        'query': tavily_response.get('query', ''),
        'results': limited_results
    }
    return format_tavily_results(limited_response)


def main():
    args = parse_arguments()
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise SystemExit("TAVILY_API_KEY is not set")
    client = TavilyClient(api_key)

    df = read_table(args.input)
    results = asyncio.run(
        process_claims_parallel(
            df, client, args.claim_column, build_search_kwargs(args), args.max_concurrent
        )
    )

    df['evidence_tavily_20_results'] = [format_tavily_results(result) for result in results]
    df['evidence_tavily_5_results'] = [extract_first_n_results_tavily(result, 5) for result in results]
    df["tavily_results_json"] = [json.dumps(result) if result else "{}" for result in results]

    failed = sum(1 for result in results if result.get("error"))
    print(f"Done: {len(results)} claims, {failed} failed")
    write_table(df, args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
