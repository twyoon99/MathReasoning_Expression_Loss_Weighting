import re


def extract_math_expressions(text):
    matches = []
    latex_patterns = [
        (r'\$([^\n]+?)\$', 'inline'),
        (r'\\\[(.+?)\\\]', 'display_brackets'),
        (r'\$\$(.+?)\$\$', 'display_dollars'),
        (r'\\begin\{(align\*?|equation\*?|gather\*?)\}(.+?)\\end\{\1\}', 'env')
    ]
    for pattern, label in latex_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1) if label != 'env' else match.group(2)
            start, end = match.start(), match.end()
            if label == 'inline':
                alphabetic_count = sum(c.isalpha() for c in content)
                if alphabetic_count / len(content) > 0.5:
                    continue
            matches.append((start, end, content))
    text_math_pattern = r'(?<![A-Za-z0-9])(?:(?:\d+(?:\.\d+)?)|[A-Za-z])\b(?:[^\S\r\n]*(?:[=+\-*/\u00F7\u00D7\u00B7^%<>!]+)[^\S\r\n]*\b(?:(?:\d+(?:\.\d+)?)|[A-Za-z])\b)+(?![A-Za-z0-9])'
    for match in re.finditer(text_math_pattern, text):
        matches.append((match.start(), match.end(), match.group(0)))
    boxed_pattern = r'(\\boxed\{.*?\})'
    for match in re.finditer(boxed_pattern, text):
        matches.append((match.start(), match.end(), match.group(0)))
    number_pattern = r'\b\d+(?:\.\d+)?\b'
    for match in re.finditer(number_pattern, text):
        matches.append((match.start(), match.end(), match.group(0)))
    matches = sorted(matches, key=lambda x: (x[0], -(x[1] - x[0])))
    final_matches = []
    last_end = -1
    for start, end, content in matches:
        if start >= last_end:
            final_matches.append((start, end, content))
            last_end = end
    return final_matches

def extract_boxed_contents(text):
    boxed_pattern = r'\\boxed\{(.*?)\}'
    matches = list(re.finditer(boxed_pattern, text))
    return [(match.start(1), match.end(1), match.group(1)) for match in matches if match.group(1) != '']

def mark_math_tokens(text, offsets, math_spans):
    math_mask = [False] * len(offsets)
    for i, (start, end) in enumerate(offsets):
        for m_start, m_end, _ in math_spans:
            if start < m_end and end > m_start:
                math_mask[i] = True
                break
    return math_mask