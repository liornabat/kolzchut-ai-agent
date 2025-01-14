import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

db_table = os.getenv("DB_TABLE", "site_pages_2")
docs_locations = os.getenv("DOCS_DIRECTORY", "./source_markdown")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
@dataclass
class ProcessedChunk:
    file_path: str  # Changed from url to file_path
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks


async def get_title_and_summary(chunk: str, file_path: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """אתה עוזר בינה מלאכותית שמחלצת כותרות ותקצירים מקטעי תיעוד.
    החזר אובייקט JSON עם המפתחות 'title' ו-'summary'.
    עבור הכותרת: אם זה נראה כמו תחילת מסמך, חלץ את הכותרת שלו. אם זה קטע אמצעי, צור כותרת תיאורית.
    עבור התקציר: צור סיכום תמציתי של הנקודות העיקריות בקטע זה.
    שמור על הכותרת והתקציר תמציתיים אך אינפורמטיביים.
    חשוב: יש להשתמש בעברית בלבד."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"File: {file_path}\n\nContent:\n{chunk[:5000]}..."}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 3072  # Return zero vector on error


async def process_chunk(chunk: str, chunk_number: int, file_path: str, max_retries: int = 3) -> ProcessedChunk:
    """Process a single chunk of text with retries."""
    for attempt in range(max_retries):
        try:
            # Get title and summary
            extracted = await get_title_and_summary(chunk, file_path)

            # Get embedding
            embedding = await get_embedding(chunk)

            # Create metadata
            metadata = {
                "source": docs_locations,
                "chunk_size": len(chunk),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "file_path": str(file_path),
                "attempt": attempt + 1
            }

            return ProcessedChunk(
                file_path=str(file_path),
                chunk_number=chunk_number,
                title=extracted['title'],
                summary=extracted['summary'],
                content=chunk,
                metadata=metadata,
                embedding=embedding
            )
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to process chunk after {max_retries} attempts: {str(e)}")
                raise
            await asyncio.sleep(1)  # Wait before retry


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.file_path,  # Using file_path as URL for consistency
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table(db_table).insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.file_path}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None


async def process_and_store_document(file_path: Path, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)

    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, file_path)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)


def get_markdown_files(directory: str) -> List[Path]:
    """Get all markdown files from the specified directory and its subdirectories."""
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"Directory not found: {directory}")
        return []

    markdown_files = []
    for file_path in directory_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.md', '.mdx']:
            markdown_files.append(file_path)

    return markdown_files


async def get_unprocessed_files(files: List[Path]) -> List[Path]:
    """Filter out files that have already been processed in the database."""
    try:
        # Query Supabase for existing file paths
        response = supabase.table(db_table) \
            .select("url") \
            .execute()

        # Extract unique file paths from the response
        processed_files = set(item['url'] for item in response.data)

        # Filter out files that are already in the database
        unprocessed_files = [file for file in files if str(file) not in processed_files]

        print(f"Total files: {len(files)}")
        print(f"Already processed: {len(processed_files)}")
        print(f"New files to process: {len(unprocessed_files)}")

        return unprocessed_files
    except Exception as e:
        print(f"Error checking processed files: {str(e)}")
        return files  # Return all files if there's an error


async def process_files_parallel(files: List[Path], max_concurrent: int = 5):
    """Process multiple files in parallel with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_file(file_path: Path):
        try:
            async with semaphore:
                print(f"Processing file: {file_path}")
                content = file_path.read_text(encoding='utf-8')
                await process_and_store_document(file_path, content)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    try:
        # Process files in smaller batches
        batch_size = 10
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} files)")
            await asyncio.gather(*[process_file(file) for file in batch])

            # Add a small delay between batches
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")


async def main():
    try:


        # Get all markdown files
        markdown_files = get_markdown_files(docs_locations)
        if not markdown_files:
            print("No markdown files found")
            return

        print(f"Found {len(markdown_files)} markdown files")

        # Filter out already processed files
        unprocessed_files = await get_unprocessed_files(markdown_files)
        if not unprocessed_files:
            print("All files have already been processed")
            return

        # Process files in parallel
        await process_files_parallel(unprocessed_files, max_concurrent=10)

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())