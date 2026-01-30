+ Problem (unicode1): Understanding Unicode
(a) What Unicode character does chr(0) return?
+ chr(0) returns the Unicode null character (NUL).
(b) How does this character’s string representation (__repr__()) differ from its printed representation?
(b) The repr() representation displays the escape sequence '\x00', while printing it produces no visible output.
(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
(c) When printed, the null character is invisible and does not disrupt the visible text flow, though it remains part of the string.






+ Problem (unicode2): Unicode Encodings
(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
(a) UTF-8 is preferred because it is backward compatible with ASCII, ensuring efficient encoding for common English text, and it is the dominant encoding for web and storage systems, making training data more representative. Additionally, UTF-8's variable-length encoding allows compact representation of multilingual text, unlike UTF-16 or UTF-32, which use fixed widths that waste space for ASCII characters.
(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
(b) The function produces incorrect output for the byte string b'\xe2\x82\xac' (the UTF-8 encoding of "€") because it attempts to decode each byte individually, but UTF-8 characters can span multiple bytes, leading to decoding errors or exceptions.
(c) Give a two byte sequence that does not decode to any Unicode character(s).
Deliverable: An example, with a one-sentence explanation.

(c) The two-byte sequence b'\xc0\x80' does not decode to any Unicode character because it is an invalid overlong encoding of the null character, which is disallowed in UTF-8.