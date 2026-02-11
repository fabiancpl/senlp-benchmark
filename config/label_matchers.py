"""Label matchers.

Inventory of class labels for each task. For binary and multi-class tasks, the
object represents the matching between ID and class in natural text. For
multi-label, this represents the list of independent labels. For NER, the
possible classes to be detected for the LLM. And for MLM, the POS tags used
during the data collector stage to mask only VERBS.
"""

matchers = {
    "bug_issue": {
        0: "not a bug",
        1: "bug"
    },
    "closed_question": {
        0: "open",
        1: "not a real question",
        2: "off topic",
        3: "not constructive",
        4: "too localized"
    },
    "comment_type_python": [
        "usage",
        "parameters",
        "developmentNotes",
        "expand",
        "Summary"
    ],
    "comment_type_java": [
        "summary",
        "ownership",
        "expand",
        "usage",
        "pointer",
        "deprecation",
        "rational"
    ],
    "comment_type_pharo": [
        "keyimplementationpoints",
        "example",
        "responsibilities",
        "classreferences",
        "intent",
        "keymessages",
        "collaborators"
    ],
    "commit_intent": {
        0: "other",
        1: "perfective",
        2: "corrective"
    },
    "functional_requirement": {
        0: "non-functional",
        1: "functional"
    },
    "incivility": {
        0: "uncivil",
        1: "civil"
    },
    "issue_intention": {
        0: "aspect evaluation",
        1: "feature request",
        2: "information giving",
        3: "information seeking",
        4: "others",
        5: "problem discovery",
        6: "solution proposal"
    },
    "issue_type": {
        0: "bug",
        1: "enhancement",
        2: "question"
    },
    "quality_requirement": {
        0: "non-quality",
        1: "quality"
    },
    "question_quality": {
        0: "LQ_CLOSE",
        1: "LQ_EDIT",
        2: "HQ"
    },
    "requirement_completion": [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SPACE",
        "SYM",
        "VERB",
        "X"
    ],
    "review_aspect": [
        "usability",
        "others",
        "onlysentiment",
        "bug",
        "performance",
        "community",
        "documentation",
        "compatibility",
        "legal",
        "portability",
        "security"
    ],
    "review_type": {
        0: "information seeking",
        1: "information giving",
        2: "feature request",
        3: "problem discovery"
    },
    "safety_issue": {
        0: "NO",
        1: "YES"
    },
    "se_entities": [
        "O",
        "B-Data_Structure", "I-Data_Structure",
        "B-Application", "I-Application",
        "B-Code_Block", "I-Code_Block",
        "B-Function", "I-Function",
        "B-Data_Type", "I-Data_Type",
        "B-Language", "I-Language",
        "B-Library", "I-Library",
        "B-Variable", "I-Variable",
        "B-Device", "I-Device",
        "B-User_Name", "I-User_Name",
        "B-User_Interface_Element", "I-User_Interface_Element",
        "B-Class", "I-Class",
        "B-Website", "I-Website",
        "B-Version", "I-Version",
        "B-File_Name", "I-File_Name",
        "B-File_Type", "I-File_Type",
        "B-Operating_System", "I-Operating_System",
        "B-Output_Block", "I-Output_Block",
        "B-Algorithm", "I-Algorithm",
        "B-HTML_XML_Tag", "I-HTML_XML_Tag"
    ],
    "security_requirement": {
        0: "nonsec",
        1: "sec"
    },
    "sentiment": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "smell_doc": [
        "fragmented",
        "tangled",
        "excessive",
        "bloated",
        "lazy"
    ],
    "story_points": None,
    "tone_bearing": {
        0: "technical",
        1: "non-technical",
    }
}
