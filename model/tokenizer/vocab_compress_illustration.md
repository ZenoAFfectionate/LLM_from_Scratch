# Vocabulary Compression - Detailed Explanation

## Overview

Vocabulary compression reduces the effective vocabulary size by mapping semantically equivalent tokens to the same canonical ID. This is particularly useful for N-gram language modeling where we want to treat variations of the same word as the same entity.

## The Normalization Pipeline

The normalization pipeline consists of 9 sequential steps. Each step transforms the text in a specific way:

```python
self.normalizer = normalizers.Sequence([
    normalizers.NFKC(),                          # Step 1
    normalizers.NFD(),                           # Step 2
    normalizers.StripAccents(),                  # Step 3
    normalizers.Lowercase(),                     # Step 4
    normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),  # Step 5
    normalizers.Replace(Regex(r"^ $"), SENTINEL),    # Step 6
    normalizers.Strip(),                             # Step 7
    normalizers.Replace(SENTINEL, " "),              # Step 8
])
```

### Step-by-Step Breakdown

#### **Step 1: NFKC() - Unicode Compatibility Normalization**

**What it does:** Converts characters to their compatibility equivalents and composes them.

**Examples:**
- `①` (circled digit one) → `1` (normal digit)
- `ﬁ` (ligature fi) → `fi` (separate f + i)
- `²` (superscript 2) → `2` (normal digit)
- `Ⅳ` (Roman numeral four) → `IV`
- `㎡` (square meter symbol) → `m2`

**Why it's useful:** Many Unicode characters have "fancy" visual representations but represent the same concept. This normalizes them to their standard forms.

---

#### **Step 2: NFD() - Canonical Decomposition**

**What it does:** Decomposes characters into their base character + combining diacritical marks.

**Examples:**
- `é` (single character U+00E9) → `e` + `´` (base + combining acute accent)
- `ñ` → `n` + `~` (base + combining tilde)
- `ü` → `u` + `¨` (base + combining diaeresis)

**Why it's useful:** This separates the base character from accents, preparing them for the next step (accent removal). It's crucial for handling international text.

---

#### **Step 3: StripAccents() - Remove Diacritical Marks**

**What it does:** Removes all combining diacritical marks (accents, tildes, umlauts, etc.).

**Examples:**
- `e` + `´` → `e` (removes acute accent)
- `n` + `~` → `n` (removes tilde)
- `café` → `cafe`
- `naïve` → `naive`
- `Zürich` → `Zurich`

**Why it's useful:** Treats accented and non-accented versions of words as the same. In many contexts, "cafe" and "café" should be considered equivalent.

---

#### **Step 4: Lowercase() - Convert to Lowercase**

**What it does:** Converts all uppercase letters to lowercase.

**Examples:**
- `Apple` → `apple`
- `HELLO` → `hello`
- `WoRlD` → `world`
- `OpenAI` → `openai`

**Why it's useful:** Makes tokenization case-insensitive. "Apple", "apple", and "APPLE" all refer to the same concept and should share statistics.

---

#### **Step 5: Replace(Regex(r"[ \t\r\n]+"), " ") - Normalize Whitespace**

**What it does:** Replaces any sequence of whitespace characters (spaces, tabs, newlines, carriage returns) with a single space.

**Examples:**
- `"hello  world"` → `"hello world"` (multiple spaces → single space)
- `"hello\tworld"` → `"hello world"` (tab → space)
- `"hello\n\nworld"` → `"hello world"` (newlines → space)
- `"  multiple   spaces  "` → `" multiple spaces "` (normalized but edges preserved)

**Why it's useful:** Different tokenizers might use different whitespace conventions (spaces vs tabs). This normalizes them all.

---

#### **Step 6: Replace(Regex(r"^ $"), SENTINEL) - Protect Standalone Spaces**

**What it does:** Replaces a string that is ONLY a single space with a sentinel character (U+E000, a private use Unicode character).

**Examples:**
- `" "` (just a space) → `"\uE000"` (sentinel)
- `" hello"` → `" hello"` (unchanged, not just a space)
- `"  "` → `" "` → `"\uE000"` (first normalized to single space, then protected)

**Why it's useful:** The next step (Strip) would remove standalone spaces entirely. We need to preserve them because space tokens are important in the vocabulary. The sentinel acts as a temporary placeholder.

---

#### **Step 7: Strip() - Remove Leading/Trailing Whitespace**

**What it does:** Removes all whitespace from the beginning and end of the string.

**Examples:**
- `"  hello  "` → `"hello"`
- `" apple"` → `"apple"`
- `"world  "` → `"world"`
- `"\uE000"` → `"\uE000"` (sentinel is preserved because it's not whitespace)

**Why it's useful:** Tokens like ` "apple"` (with leading space) and `"apple"` should be treated as the same word. This removes the distinction based on leading/trailing whitespace.

**Note:** This is why Step 6 was necessary - without the sentinel, a token that is ONLY a space would become an empty string here.

---

#### **Step 8: Replace(SENTINEL, " ") - Restore Standalone Spaces**

**What it does:** Converts the sentinel character back to a space.

**Examples:**
- `"\uE000"` → `" "` (restore the space)
- `"hello"` → `"hello"` (unchanged, no sentinel)

**Why it's useful:** Now that we've safely passed the Strip() step, we restore standalone space tokens to their proper form.

---

## Complete Example Walkthrough

Let's trace the token `" Café "` (with leading/trailing spaces and accent) through the entire pipeline:

```
Original:     " Café "
After NFKC:   " Café "       (no change, é already in compatibility form)
After NFD:    " Cafe´ "      (é decomposed to e + combining acute accent)
After Strip:  " Cafe "       (accent mark removed)
After Lower:  " cafe "       (C → c)
After WSNorm: " cafe "       (single spaces remain)
After Sent:   " cafe "       (not just a space, so no sentinel)
After Strip:  "cafe"         (leading/trailing spaces removed)
After Unsent: "cafe"         (no sentinel to restore)

Final normalized form: "cafe"
```

Now compare with `"CAFE"`:

```
Original:     "CAFE"
After NFKC:   "CAFE"
After NFD:    "CAFE"
After Strip:  "CAFE"
After Lower:  "cafe"
After WSNorm: "cafe"
After Sent:   "cafe"
After Strip:  "cafe"
After Unsent: "cafe"

Final normalized form: "cafe"
```

**Both map to the same canonical form: `"cafe"`**

This means in the compressed vocabulary, tokens with IDs representing `" Café "`, `"CAFE"`, `"cafe"`, `" cafe"`, etc., all get mapped to the same compressed ID!

## Impact on Language Modeling

### Before Compression:
- Token ID 5432: `"Apple"` - seen 100 times
- Token ID 8921: `" apple"` - seen 150 times
- Token ID 2341: `"APPLE"` - seen 20 times

These are treated as completely different tokens with separate statistics.

### After Compression:
- All three tokens → Compressed ID 1523
- Combined count: 270 occurrences

The model can now learn better statistics because it pools together all variations of the same word.

## Real Results from Your Datasets

### TinyStories:
- **Original vocabulary:** 10,000 tokens
- **Compressed vocabulary:** 8,098 tokens
- **Compression ratio:** 19.02%
- **Interpretation:** About 1,902 tokens were duplicates with different formatting

### OpenWebText:
- **Original vocabulary:** 32,000 tokens
- **Compressed vocabulary:** 22,779 tokens
- **Compression ratio:** 28.82%
- **Interpretation:** About 9,221 tokens (nearly 29%!) were variations of other tokens

## Why This Matters for N-gram Models

In N-gram language modeling, we're counting sequences of tokens. If we have:
- Bigram: ("I", "love") → "apple" appears 10 times
- Bigram: ("I", "love") → "Apple" appears 5 times

Without compression, these are separate statistics. With compression, we combine them:
- Bigram: ("i", "love") → "apple" appears 15 times

This leads to:
1. **Better probability estimates** (more data per canonical form)
2. **Reduced sparsity** (fewer unique N-grams to track)
3. **Better generalization** (model learns semantic patterns, not formatting patterns)

## Code Implementation

The lookup table is simply a numpy array where `lookup_table[old_id] = new_id`:

```python
# Example lookup table (first 20 entries from TinyStories)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 9, 12, 13, 14, 15, ...]
#                                  ↑           ↑
#  Token ID 10 maps to ID 9 ------┘           │
#  Token ID 13 also maps to ID 9 ────────────┘
# These two different original tokens normalized to the same form!
```

## Usage in Your Code

When you have a sequence of token IDs from the BPE tokenizer:

```python
original_ids = [5432, 8921, 2341, 9876]  # [Apple, apple, APPLE, world]

# Compress to canonical IDs
compressed_tokenizer = CompressedTokenizer(vocab_file, merges_file)
canonical_ids = compressed_tokenizer(original_ids)
# Result: [1523, 1523, 1523, 7654]  # All "apple" variants → same ID
```

Now you can build N-gram counts using the canonical IDs, which gives you better statistics!
