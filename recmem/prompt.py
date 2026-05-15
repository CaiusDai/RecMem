import datetime


ANSWER_PROMPT_LOCOMO = """
You are an intelligent memory assistant tasked with answering questions using conversation memories.

# CONTEXT
You have access to memories from two speakers in a conversation. These memories are timestamped and may be relevant to the question.

There are three types of memories:
1. Episodic Memories: refined summaries of related conversation turns about the same topic.
2. Semantic Memories: concise, fact-like pieces extracted from conversations.
3. Raw Memories: unprocessed conversation snippets between the two speakers.

Context Rules:
1. Episodic and semantic memories may overlap in content (event summary vs. atomic facts). Avoid double-counting redundant evidence.
2. Carefully analyze all three memory types and identify information that is actually useful for answering the question.
3. Memories within each type are sorted by relevance.

# INSTRUCTIONS
1. Carefully read all provided memories.
2. Pay close attention to timestamps when time is relevant.
3. If the question asks about a specific event or fact (who / where / when / what), look for direct, explicit evidence in the memories.
4. If the question asks for advice, recommendations, or what kind of response the user would prefer, 
   - first identify any user-specific preferences, habits, constraints, or past actions from the memories, 
   - then base your suggestion primarily on these user-specific signals, 
   - and only fall back to generic advice when no relevant user information exists.
5. If memories contain contradictory information, prioritize the most recent memory.
6. For time references (e.g., "last year", "two months ago"), convert them into concrete dates based on the memory timestamp. 
   For example, if a memory from 4 May 2022 says "went to India last year", infer that the trip happened in 2021, 
   and answer with "2021" or "the year before 2022", not just "last year".
7. In raw memories, the final timestamp marks the conversation time. Do not confuse the time of conversation with the time when an event actually happened if the text distinguishes them.
8. Do not say "no information found" if there are related memories that can reasonably guide a personalized answer. 
   Only abstain when there is truly no relevant evidence.


# APPROACH (Think step by step internally)
1. Identify whether the question is (a) a factual query or (b) an advice / preference / recommendation query.
2. Retrieve all memories that are clearly related to the question.
3. Check timestamps and content to locate the most reliable and up-to-date information.
4. For factual queries, pinpoint explicit mentions of dates, times, locations, entities, or events that directly answer the question.
5. For advice / preference queries, determine what the user has already done, bought, liked, disliked, or constrained, and use these as anchors for a tailored answer.
6. If temporal reasoning or simple calculation is needed, do it internally and convert the result into a concrete, explicit date or time span in the final answer.
7. Formulate a precise, concise answer that directly addresses the question and is fully supported by the memories and reasonable inferences from them.
8. The answer should be less than 5-6 words.

Episodic Memories: {{ episodic_memories }}
Raw Memories: {{ raw_memories }}
Semantic Memories: {{ semantic_memories }}
Question: {{ question }}
Answer:
"""

ANSWER_PROMPT_LME = """
You are an intelligent memory assistant tasked with answering questions using conversation memories.

# CONTEXT
You have access to memories from two speakers in a conversation. These memories are timestamped and may be relevant to the question.

There are three types of memories:
1. Episodic Memories: refined summaries of related conversation turns about the same topic.
2. Semantic Memories: concise, fact-like pieces extracted from conversations.
3. Raw Memories: unprocessed conversation snippets between the two speakers.

Context Rules:
1. Episodic and semantic memories may overlap in content (event summary vs. atomic facts). Avoid double-counting redundant evidence.
2. Carefully analyze all three memory types and identify information that is actually useful for answering the question.
3. Memories within each type are sorted by relevance.

# INSTRUCTIONS
1. Carefully read all provided memories.
2. Pay close attention to timestamps when time is relevant.
3. If the question asks about a specific event or fact (who / where / when / what), look for direct, explicit evidence in the memories.
4. If the question asks for advice, recommendations, or what kind of response the user would prefer, 
   - first identify any user-specific preferences, habits, constraints, or past actions from the memories, 
   - then base your suggestion primarily on these user-specific signals, 
   - and only fall back to generic advice when no relevant user information exists.
5. If memories contain contradictory information, prioritize the most recent memory.
6. For time references (e.g., "last year", "two months ago"), convert them into concrete dates based on the memory timestamp. 
   For example, if a memory from 4 May 2022 says "went to India last year", infer that the trip happened in 2021, 
   and answer with "2021" or "the year before 2022", not just "last year".
7. In raw memories, the final timestamp marks the conversation time. Do not confuse the time of conversation with the time when an event actually happened if the text distinguishes them.
8. Do not say "no information found" if there are related memories that can reasonably guide a personalized answer. 
   Only abstain when there is truly no relevant evidence.

# APPROACH (Think step by step internally)
1. Identify whether the question is (a) a factual query or (b) an advice / preference / recommendation query.
2. Retrieve all memories that are clearly related to the question.
3. Check timestamps and content to locate the most reliable and up-to-date information.
4. For factual queries, pinpoint explicit mentions of dates, times, locations, entities, or events that directly answer the question.
5. For advice / preference queries, determine what the user has already done, bought, liked, disliked, or constrained, and use these as anchors for a tailored answer.
6. If temporal reasoning or simple calculation is needed, do it internally and convert the result into a concrete, explicit date or time span in the final answer.
7. Formulate a precise, concise answer that directly addresses the question and is fully supported by the memories and reasonable inferences from them.

Episodic Memories: {{ episodic_memories }}
Raw Memories: {{ raw_memories }}
Semantic Memories: {{ semantic_memories }}
Question: {{ question }}
Answer:
"""


EPISODIC_MERGE_PROMPT = """
You are an Episodic Memory Merger for a long-term memory system.

Global principles:
- Be user-centric: focus on the real people who are having the conversation (the main user and the assistant or other named speakers).
- Prioritize their real-world goals, projects, constraints, decisions, and emotional states.
- Treat fictional, hypothetical, or example scenarios mentioned inside the conversation as background context. Only consider them central if they clearly matter for the speakers' own situation.

Goal:
You are given exactly TWO inputs:
- one NEW memory piece, derived from a recent conversation snippet,
- one PAST episodic memory piece, previously stored.
Your job is to decide whether these two pieces describe the same ongoing topic or episode, and if so, produce ONE updated episodic memory that integrates both.

Your tasks:
1. Decide whether the new memory piece and the past memory piece describe the same ongoing topic or episode for the conversation participants.
2. If yes, merge them into ONE coherent episodic memory and return the merged result.

--------------------
When to merge
--------------------
Treat the new and past memory pieces as strongly related (should_merge = "yes") if MOST of the following hold:
- They describe the same ongoing situation, goal, project, problem, or life event for the same main person(s) in the conversation (user, assistant, or other real participants), not just similar fictional stories or generic examples.
- The new piece clearly adds follow-up details, clarifications, corrections, or outcomes to what the past piece already describes, instead of starting a new, separate topic.
- A human reader, after reading only these two pieces, would naturally describe them as different parts of one story about those conversation participants, not as two unrelated stories.

Special case: new piece is only from the assistant
- Sometimes the new memory piece mainly contains a long answer or explanation from the assistant, with little or no user text.
- If this assistant-only piece mostly consists of generic background information, universal tips, or a long example/story that could apply to many people, and it does NOT clearly update or resolve the specific situation in the past memory, you should NOT merge.
- You may still merge an assistant-only piece when it explicitly continues the same concrete situation as in the past memory (for example, by referring to the same plan, constraint, or previous decision of the user, and by providing the next step, revision, or outcome for that situation).

If the two pieces only share superficial elements (e.g., they mention the same domain or concept but refer to different user problems or different episodes), or if their overlap is mainly about similar example stories, hypothetical scenarios, or generic explanations that do not change the speakers' own situation, you should NOT merge.

If you are uncertain whether they refer to the same ongoing topic, prefer "no" for should_merge.

--------------------
How to merge (if related)
--------------------
If you decide to merge:
- Preserve the important details from BOTH the past memory and the new memory.
- Organize the merged memory in temporal order so that the narrative is easy to follow.
- Emphasize how the speakers' situation, plans, beliefs, or preferences evolve over time, rather than reproducing long internal stories or technical explanations.
- Make the logical relation explicit using phrases like "later", "afterwards", "as a result", "eventually", etc.
- Keep the merged memory compact. Do not copy long fictional plots, full technical explanations, or long lists; refer to them briefly if needed (e.g., "the assistant told a story about X to illustrate Y").
- Handle time expressions carefully:
  - Keep numeric timestamps or explicit dates as the ground truth (do not change their values).
  - Rewrite vague relative time words like "yesterday", "last week", "tomorrow", "next month", "in two days"
    into anchored expressions relative to a date mentioned in the input, for example:
    "the day before 2 January 2024", "two days after 2025-03-10",
    "later in the same week as 2025-04-01".
  - You do NOT need to compute calendar arithmetic (e.g., you do not need to turn
    "the day before 2 January 2024" into "1 January 2024").
- Do NOT invent new events or details that are not supported by either piece.

The merged memory should remain a single, coherent paragraph of episodic narrative, not a list of bullet points.

--------------------
Output format
--------------------
Return ONLY a JSON object of the form:

{
  "should_merge": "yes" or "no",
  "merged_memory": "merged memory piece if should_merge is yes, otherwise an empty string"
}
Now process the following input.


<New Memory Piece>:
{{new_memory}}

<Past Memory Piece>:
{{past_memory}}

Output:
"""


EPISODIC_GENERATION_PROMPT = """

You are an Episodic Memory Constructor for a long-term memory system used by an LLM-based agent.

Your input:
- A list of chat message turns separated by '-'. Each chat message turn has:
  - One message turn between two speakers (contains one message per speaker),
  - a timestamp (when the message turn was written),
  - the message text.
- All messages in the input are loosely related to one broad theme, but they may contain several distinct subtopics and span multiple days or weeks.

Your task:
- Convert these messages into a small number of coherent episodic memories ("episodes").
- Each episode should describe how ONE subtopic evolves over time across multiple messages and timestamps.

Global principles:
- Be user-centric: focus on the real people who are actually having the conversation (the main user and the assistant / other named speakers).
- Prioritize what these speakers are doing, planning, feeling, or learning, rather than the internal details of stories, examples, or long explanations they mention.
- Treat fictional, hypothetical, or example scenarios (stories inside the conversation, sample dialogues, illustrative anecdotes) as background. Only include them if they clearly reveal something about the speakers' own preferences, goals, constraints, or knowledge.

====================
INSTRUCTIONS (THINK STEP BY STEP)
====================

1. Identify topic threads
- First, mentally group the messages into topic threads.
- Messages belong to the same thread if they refer to the same ongoing goal, project, problem, or situation for the conversation participants, even if they are days or weeks apart.
- Ignore or down-weight one-off fictional or hypothetical stories that do not affect the speakers' own situation.
- If the input contains multiple independent threads, you may produce one episode per thread.
- Prefer 1–3 episodes in total. Do NOT create many tiny episodes.

2. Build temporal structure for each episode
- For each thread that has enough information, order the relevant events chronologically.
- Highlight how the situation develops over time: initial situation, updates, changes of plan, decisions, outcomes, and reflections.
- Emphasize how the speakers' state (plans, preferences, beliefs, emotional reactions) evolves, not the blow-by-blow content of long stories or explanations.
- Use connective phrases like "Initially", "Later", "Afterwards", "Eventually", "As a result" to make the temporal progression explicit.
- Whenever possible, each episode should summarize information from at least TWO different message timestamps. Avoid episodes that just restate a single message.

3. Handle time expressions correctly
- The timestamp of a message is the time when the user said it. It is NOT always the time when the described event happened.
- If a message uses relative time expressions such as "yesterday", "two days ago", "next week", rewrite them in your episode as explicit expressions relative to the timestamp. For example:
  - "the day before 2025-02-03"
  - "two days after 2025-05-10"
  - "later in the same week as 2025-03-01"
- You do NOT need to compute calendar dates (e.g., you do not need to turn "the day before 2025-02-03" into "2025-02-02").
- IMPORTANT: In your final output, NEVER use vague relative time expressions without an anchor. Avoid words like "yesterday", "tomorrow", "last week", "next week", or "in two days" unless you clearly specify what timestamp or date they are relative to.
- It is acceptable to use coarse ranges such as "later that week" or "over the following days after 2025-03-01" as long as the temporal order is unambiguous.

4. Focus on episodic narratives, not isolated facts
- Your goal is to construct narrative episodes: what happened, how it evolved, and why it matters to the user.
- Focus on the outer conversation between the two speakers (their goals, decisions, preferences, constraints, and what has been explained to them).
- When a speaker tells a long story, gives an extended example, or pastes long external text, summarize this very compactly (e.g., "the assistant told a story about X to illustrate Y") unless the detailed content is central to the speakers' own situation.
- Do NOT try to list every long-term preference or stable fact about the user. A separate semantic memory extractor will handle persistent facts and knowledge.
- You may mention important preferences or facts inside an episode if they are relevant to understanding the story, but you do not need to optimize for completeness.

5. Style and output format
- Write each episode as a short, well-formed paragraph (3 to 6 sentences) in clear, neutral language.
- Keep episodes compact. Do not reproduce long fictional plots, full technical explanations, or long lists; refer to them briefly if needed.
- Prefer merging related events into a single episode over splitting them into many small episodes.
- If there is not enough coherent information for a given subtopic, you may skip creating an episode for it.

====================
OUTPUT FORMAT (JSON)
====================

Return ONLY a JSON object in the following format:

{
  "episodes": [
    "First episodic memory paragraph...",
    "Second episodic memory paragraph..."
  ]
}

If you cannot form any meaningful episode, return:

{
  "episodes": []
}

Messages:
{{raw_conversation}}

Output:
"""

SEMANTIC_EXTRACTION_PROMPT = """
You are a Semantic Memory Extractor for a long-term memory system.

Inputs:
- Raw reference R: the original detailed messages that are related
- Episodic memory E: a short narrative summary generated from the raw reference R.
- Old semantic memories S: previously stored facts about the topic related to E.

Goal:
Extract NEW, HIGH-UTILITY facts that will help answer future questions about the users.
*IMPORTANT*: Prioritize concrete details from R that are missing or strongly compressed in E.

Core principles:
1) USER-CENTRIC: Focus on the user's goals, preferences, constraints, decisions,
   actions, and recurring plans.
2) TEMPORALLY GROUNDED: For events and changes, include an explicit date anchor
   when available.
3) COMPACT BUT USEFUL: Output less than 10 facts. Fewer, higher-value facts
   are better than many low-value facts.

Each fact MUST belong to one of these types:

1) USER_EVENT
   - The user asked for something, attended something, started or stopped
     something, or made a concrete decision.
   - Include a date anchor from R if possible, e.g., "On 2023-05-22, the user ...".

2) USER_CONSTRAINT_OR_PREFERENCE
   - A relatively stable preference, constraint, or recurring plan
     (e.g., long-term goals, platform choice, budget range, time constraints,
      content or style preferences).

3) TIME_ANCHORED_UPDATE
   - A change in behavior, preferences, tools, roles, budgets, relationships, etc.
   - State the new value and, if known, the old value or state, with a clear time anchor.

4) ENTITY_RELATION (USER-RELEVANT)
   - A specific relationship between named entities that is relevant to the user
     (e.g., the user's roles, organizations, projects, courses, tools, locations,
     or other people they interact with).
   - Only keep such a fact if it is specific and likely to matter in future
     reasoning about the user.
   - Do NOT store generic world knowledge that is not grounded in the user.

Do NOT output generic best-practice tips, long checklists, or broad explanations
unless they are clearly framed as what this user has done, is doing, or plans to do.

Time handling
 - When expressing time, use explicit or anchored expressions such as:
   - "in March 2025", "on 2025-03-10",
   - "two days after 2025-03-10",
   - "later in the same week as 2025-04-01".
 - You do NOT need to compute calendar arithmetic; describing offsets like "two days after <date>" is fine.
 - IMPORTANT: In the final facts, NEVER use vague, unanchored relative time words such as:
   "yesterday", "tomorrow", "last week", "next week", "next month", "in two days"
   unless you clearly specify what date or event they are relative to.

Novelty and updates:
- A fact is NOT new if it expresses the same meaning as any item in S,
  even with different wording.
- Keep a fact only if it is specific, self-contained (no ambiguous pronouns),
  plausibly useful for future reasoning, and not semantically equivalent to S.
- If a fact refines, updates, or contradicts an item in S (e.g., preference,
  role, tool, relationship, or budget change), keep it as a TIME_ANCHORED_UPDATE
  that clearly states the change.

If you have more than 8–10 candidate facts, select those that are:
- Most time-anchored,
- Most clearly missing or compressed in E,

Style:
- One fact per sentence.
- Neutral, factual tone.
- Do NOT speculate beyond what E, R, and S support.
- Avoid long lists; summarize them into a single concise fact when possible.

Output format:
Return ONLY a JSON object:

{
  "facts": [
    "First new semantic fact...",
    "Second new semantic fact..."
  ]
}

If there are no new facts, return:

{
  "facts": []
}

Episodic Memory:
{{episodic_memory}}

Raw Reference:
{{raw_reference}}

Old Semantic Memories:
{{old_semantic_memories}}
"""


SEMANTIC_EXTRACTION_PROMPT_ABLATION = """
You are a Semantic Memory Extractor for a long-term memory system.

Inputs:
- Raw reference R: the original detailed messages that are related

Goal:
Extract NEW, HIGH-UTILITY facts that will help answer future questions about the users.

Core principles:
1) USER-CENTRIC: Focus on the user's goals, preferences, constraints, decisions,
   actions, and recurring plans.
2) TEMPORALLY GROUNDED: For events and changes, include an explicit date anchor
   when available.
3) COMPACT BUT USEFUL: Output less than 10 facts. Fewer, higher-value facts
   are better than many low-value facts.

Each fact MUST belong to one of these types:

1) USER_EVENT
   - The user asked for something, attended something, started or stopped
     something, or made a concrete decision.
   - Include a date anchor from R if possible, e.g., "On 2023-05-22, the user ...".

2) USER_CONSTRAINT_OR_PREFERENCE
   - A relatively stable preference, constraint, or recurring plan
     (e.g., long-term goals, platform choice, budget range, time constraints,
      content or style preferences).

3) TIME_ANCHORED_UPDATE
   - A change in behavior, preferences, tools, roles, budgets, relationships, etc.
   - State the new value and, if known, the old value or state, with a clear time anchor.

4) ENTITY_RELATION (USER-RELEVANT)
   - A specific relationship between named entities that is relevant to the user
     (e.g., the user's roles, organizations, projects, courses, tools, locations,
     or other people they interact with).
   - Only keep such a fact if it is specific and likely to matter in future
     reasoning about the user.
   - Do NOT store generic world knowledge that is not grounded in the user.

Do NOT output generic best-practice tips, long checklists, or broad explanations
unless they are clearly framed as what this user has done, is doing, or plans to do.

Time handling
 - When expressing time, use explicit or anchored expressions such as:
   - "in March 2025", "on 2025-03-10",
   - "two days after 2025-03-10",
   - "later in the same week as 2025-04-01".
 - You do NOT need to compute calendar arithmetic; describing offsets like "two days after <date>" is fine.
 - IMPORTANT: In the final facts, NEVER use vague, unanchored relative time words such as:
   "yesterday", "tomorrow", "last week", "next week", "next month", "in two days"
   unless you clearly specify what date or event they are relative to.

If you have more than 8–10 candidate facts, select those that are:
- Most time-anchored,

Style:
- One fact per sentence.
- Neutral, factual tone.
- Do NOT speculate beyond what R supports.
- Avoid long lists; summarize them into a single concise fact when possible.

Output format:
Return ONLY a JSON object:

{
  "facts": [
    "First new semantic fact...",
    "Second new semantic fact..."
  ]
}

If there are no new facts, return:

{
  "facts": []
}

Raw Reference:
{{raw_reference}}
"""
