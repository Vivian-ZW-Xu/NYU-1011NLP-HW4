from transformers import T5TokenizerFast

def load_lines(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

# Load data
train_nl = load_lines('data/train.nl')
train_sql = load_lines('data/train.sql')
dev_nl = load_lines('data/dev.nl')
dev_sql = load_lines('data/dev.sql')

# Calculate statistics BEFORE tokenization
def calc_stats_raw(nl_lines, sql_lines):
    nl_lens = [len(line.split()) for line in nl_lines]
    sql_lens = [len(line.split()) for line in sql_lines]

    nl_vocab = set()
    for line in nl_lines:
        nl_vocab.update(line.lower().split())

    sql_vocab = set()
    for line in sql_lines:
        sql_vocab.update(line.split())

    return {
        'num': len(nl_lines),
        'nl_len': f"{sum(nl_lens) / len(nl_lens):.2f}",
        'sql_len': f"{sum(sql_lens) / len(sql_lens):.2f}",
        'nl_vocab': len(nl_vocab),
        'sql_vocab': len(sql_vocab)
    }

# Calculate statistics AFTER T5 tokenization
def calc_stats_tokenized(nl_lines, sql_lines):
    nl_tokenized_lens = []
    sql_tokenized_lens = []

    for nl in nl_lines:
        tokens = tokenizer.encode("translate English to SQL: " + nl)
        nl_tokenized_lens.append(len(tokens))

    for sql in sql_lines:
        tokens = tokenizer.encode(sql)
        sql_tokenized_lens.append(len(tokens))

    return {
        'nl_tok_len': f"{sum(nl_tokenized_lens) / len(nl_tokenized_lens):.2f}",
        'sql_tok_len': f"{sum(sql_tokenized_lens) / len(sql_tokenized_lens):.2f}"
    }

print("=" * 70)
print("TABLE 1: Statistics BEFORE preprocessing")
print("=" * 70)

train_stats = calc_stats_raw(train_nl, train_sql)
dev_stats = calc_stats_raw(dev_nl, dev_sql)

print(f"Number of examples & {train_stats['num']} & {dev_stats['num']} \\\\")
print(f"Mean sentence length & {train_stats['nl_len']} & {dev_stats['nl_len']} \\\\")
print(f"Mean SQL query length & {train_stats['sql_len']} & {dev_stats['sql_len']} \\\\")
print(f"Vocabulary size (natural language) & {train_stats['nl_vocab']} & {dev_stats['nl_vocab']} \\\\")
print(f"Vocabulary size (SQL) & {train_stats['sql_vocab']} & {dev_stats['sql_vocab']} \\\\")

print("\n" + "=" * 70)
print("TABLE 2: Statistics AFTER T5 tokenization")
print("=" * 70)

train_tok_stats = calc_stats_tokenized(train_nl, train_sql)
dev_tok_stats = calc_stats_tokenized(dev_nl, dev_sql)

print(f"Mean tokenized sentence length & {train_tok_stats['nl_tok_len']} & {dev_tok_stats['nl_tok_len']} \\\\")
print(f"Mean tokenized SQL query length & {train_tok_stats['sql_tok_len']} & {dev_tok_stats['sql_tok_len']} \\\\")
