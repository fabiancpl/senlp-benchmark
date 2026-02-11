"""ETL pre-processing pipelines per dataset.

This module specifies the pipelines to be applied to the different datasets and their respective
entities. The pipelines need to be defined using the Scikit-Learn Pipeline interface.
"""

from sklearn.pipeline import Pipeline

from .preprocessing_steps import (
    BasicPreprocessor,
    HTMLExtractor,
    Jira2MarkdownConverter,
    RegexMasker,
)


pretraining_pipelines = {
    "arxiv": Pipeline([
        ("basic", BasicPreprocessor(lower=False, remove_newlines=True)),
        ("url", RegexMasker(pattern="URL"))
    ]),
    "github": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "jira": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("jira", Jira2MarkdownConverter()),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "stackoverflow": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor()),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ])
}

evaluation_pipelines = {
    "bug_issue": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "closed_question": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "comment_type": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("url", RegexMasker(pattern="URL"))
    ]),
    "commit_intent": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "functional_requirement": Pipeline([
        ("basic", BasicPreprocessor(lower=False))
    ]),
    "incivility": Pipeline([
        ("basic", BasicPreprocessor(lower=False))
    ]),
    "issue_intention": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "issue_type": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "quality_requirement": Pipeline([
        ("basic", BasicPreprocessor(lower=False))
    ]),
    "question_quality": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=False)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "review_aspect": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "review_type": Pipeline([
        ("basic", BasicPreprocessor(lower=False))
    ]),
    "safety_issue": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "security_requirement": Pipeline([
        ("basic", BasicPreprocessor(lower=False))
    ]),
    "sentiment": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("hash", RegexMasker(pattern="HASH")),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "smell_doc": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("html", HTMLExtractor(in_markdown=True)),
        ("code", RegexMasker(pattern="CODE")),
        ("url", RegexMasker(pattern="URL")),
    ]),
    "story_points": Pipeline([
        ("basic", BasicPreprocessor(lower=False)),
        ("jira", Jira2MarkdownConverter()),
        ("html", HTMLExtractor(in_markdown=True)),
        ("url", RegexMasker(pattern="URL")),
        ("user", RegexMasker(pattern="USER"))
    ]),
    "tone_bearing": Pipeline([
        ("basic", BasicPreprocessor(lower=False))
    ])
}
