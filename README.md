# book-megatool

![](semantic_search.png)

```
curl -X POST "http://localhost:8000/search-epub/"  \
    -F "file=./americanGods.epub"    \
    -F "query=suffering on a massive scale is not easy to understand for people"    \
    -F "top_n=1"
```

```json
{
    "results": [
        {
            "similarity": 0.440757155418396,
            "section_preview": " Without individuals we see only numbers: a thousand dead, a hundred thousand dead, “casualties may rise to a million.” With individual stories, the statistics become people—but even that is a lie, for the people continue to suffer in numbers that themselves are numbing and meaningless. Look, see the child’s swollen, swollen belly, and the flies that crawl at the corners of his eyes, his skeletal limbs: will it make it easier for you to know his name, his age, his dreams, his fears? To see him from the inside? And if it does, are we not doing a disservice to his sister, who lies in the searing dust beside him, a distorted, distended caricature of a human child? And there, if we feel for them, are they now more important to us than a thousand other children touched by the same famine, a thousand other young lives who will soon be food for the flies’ own myriad squirming children? We draw our lines around these moments of pain, and remain upon our islands, and they cannot hurt us. They a"
        }
    ]
}
```