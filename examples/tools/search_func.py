# marti/examples/tools/search_func.py
import json
import asyncio
import aiohttp
import time
from typing import List, Dict, Any

import logging
logger = logging.getLogger(__name__)

def _process_search_results(response_data, queries):
    """Process search results data"""
    if "error" in response_data or "result" not in response_data:
        return ["No search results found."] * len(queries)
    
    results = response_data.get("result", [])
    all_query_results = []
    
    for i, query in enumerate(queries):
        if i >= len(results) or not results[i]:
            all_query_results.append("No search results found.")
            continue

        query_results = results[i]
        formatted_docs = []
        
        for idx, item in enumerate(query_results):
            if not isinstance(item, dict):
                continue

            content = item.get('document', {}).get('contents', '')
            content_lines = content.split("\n")
            title = content_lines[0].strip() if content_lines else "Untitled"
            text = "\n".join(content_lines[1:]).strip() if len(content_lines) > 1 else content.strip()

            title = title or "Untitled"
            text = text or "No content available."

            formatted_docs.append(f"Doc {idx+1}(Title: {title}) {text}")
        
        result_text = "\n".join(formatted_docs) if formatted_docs else "No search results found."
        all_query_results.append(result_text)
    
    return all_query_results

async def search_engine_async_old(queries, base_url, topk=3, timeout=30, max_retries=3):
    """
    Execute asynchronous search query.
    Process a single string query or a list of queries.
    """
    if isinstance(queries, str):
        query_list = [queries]
        is_single_input = True
    else:
        query_list = queries
        is_single_input = False

    # Execute search (with retry mechanism)
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"queries": query_list, "topk": topk, "return_scores": True}
                async with session.post(
                    base_url, 
                    json=payload, 
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    
                    # Process response data
                    results = _process_search_results(response_data, query_list)
                    return results[0] if is_single_input else results

        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2**attempt))  # Number avoidance
            else:
                print(f"Search failed, already tried {max_retries + 1} times: {e}")
    
    # All retries have failed.
    fallback_result = [json.dumps({"result": "Unknown API state (no response and no error message)."})] * len(query_list)
    return fallback_result[0] if is_single_input else fallback_result


async def search_engine_async(queries, base_url, topk=3, timeout=15, max_retries=2):
    """
    Execute asynchronous search query, reduce timeout and retry times
    """
    if isinstance(queries, str):
        query_list = [queries]
        is_single_input = True
    else:
        query_list = queries
        is_single_input = False

    start_time = time.time()
    
    # Execute search (reduce the number of retries and timeout time)
    for attempt in range(max_retries + 1):
        try:
            # Set shorter connection and read timeouts
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
                payload = {"queries": query_list, "topk": topk, "return_scores": True}
                async with session.post(
                    base_url, 
                    json=payload, 
                    timeout=aiohttp.ClientTimeout(total=timeout, connect=5)
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    
                    elapsed = time.time() - start_time
                    if elapsed > 10:  # Record slow requests
                        logger.warning(f"Slow search request took {elapsed:.2f}s for queries: {query_list[:3]}")
                    
                    results = _process_search_results(response_data, query_list)
                    return results[0] if is_single_input else results

        except asyncio.TimeoutError:
            logger.warning(f"Search timeout on attempt {attempt + 1} for queries: {query_list[:3]}")
        except Exception as e:
            logger.warning(f"Search error on attempt {attempt + 1}: {str(e)[:100]}")
            
        if attempt < max_retries:
            await asyncio.sleep(0.3 * (2**attempt))  # Reduce retry interval

    # All retries have failed
    elapsed = time.time() - start_time
    logger.error(f"Search failed after {elapsed:.2f}s and {max_retries + 1} attempts")
    
    fallback_result = ["Search service unavailable."] * len(query_list)
    return fallback_result[0] if is_single_input else fallback_result


async def step(observation, action, tool_parser, **kwargs):
    """
    The agent's step function.
    Processes the last LLM action, performs search if needed, and determines the next state.
    """
    search_base_url = kwargs.get("base_url", "http://127.0.0.1:8000/retrieval")
    search_top_k = kwargs.get("topk", 3)
    
    # The 'observation' is the input that *led* to 'action'.
    # The full dialogue *after* the LLM's action is:
    next_observation = observation + [action]

    done = False
    extra_logs = {"searches_performed": []} # To track searches for this step

    parser_results = tool_parser.parse_tools(action)

    if parser_results[0] == action:
        done = True
    else:
        tool_responses = []
        for parser_result in parser_results:
            if parser_result["name"] == "search":
                # Fix: Parse JSON string into dictionary
                try:
                    args_dict = json.loads(parser_result["args"])
                    query_list = args_dict.get("query_list", [])
                    
                    # Record search found extra_logs
                    extra_logs["searches_performed"].extend(query_list)
                    
                    # Perform asynchronous search
                    search_results = await search_engine_async(query_list, base_url=search_base_url, topk=search_top_k)
                    tool_response = "\n------\n".join(search_results) if isinstance(search_results, list) else search_results
                # except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                except Exception as e:
                    tool_response = f"Error parsing search arguments: {str(e)}"
                    
            elif parser_result["name"] in ['<error>', '<empty>', '<parse_error>']:
                tool_response = json.dumps(parser_results)
            else:
                tool_response = "The tool is not supported"
                
            tool_responses.append(tool_response)

        tool_context = '\n------\n'.join(tool_responses)
        tool_context = f"\n<|im_start|>user\n<tool_response>\n{tool_context}\n</tool_response><|im_end|>\n<|im_start|>assistant"

        next_observation += [tool_context]

    # The 'next_state' for the agent loop becomes the input for the *next* LLM generation
    return {
        "next_observation": next_observation, 
        "done": done,
        "sampling_params": kwargs.get("sampling_params"), # Pass through or allow modification
        "extra_logs": extra_logs
    }