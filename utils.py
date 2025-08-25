import pandas as pd
import tiktoken
import os

def save_embeddings_to_csv(df: pd.DataFrame, save_path: str):
    """Save the DataFrame containing embeddings to a CSV file."""
    df.to_csv(save_path, index=False)
    print(f"Embeddings saved to {save_path}")

def num_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]
    elif len(chunks) == 2:
        return chunks
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]

def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string

def save_metadata(metadata: pd.DataFrame, metadata_file: str):
    """Save the metadata to a CSV file."""
    try:
        if os.path.exists(metadata_file):
            existing_metadata = pd.read_csv(metadata_file)
            updated_metadata = pd.concat([existing_metadata, metadata], ignore_index=True)
        else:
            updated_metadata = metadata
        updated_metadata.to_csv(metadata_file, index=False)
        print(f"Metadata saved to {metadata_file}")
    except Exception as e:
        print(f"Error saving metadata: {e}") 