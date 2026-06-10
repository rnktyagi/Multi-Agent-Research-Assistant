from langgraph.graph import StateGraph , START , END
from typing import Annotated , TypedDict , List
from pydantic import BaseModel , Field
import operator
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from src.tools import web_search_tool
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(model="openai/gpt-4o-mini",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
                max_tokens=10000)

class PlannerState(BaseModel) :
    objectives : List[str] = Field(..., description="The objectives for the given user input to conduct a research on.")

class ResearchState(BaseModel) :
    source : List[str] = Field(...,description="The list of all sources the research is conducted from.")
    content : List[str] = Field(...,description="The actual content after the research is conducted by the agent.")

class SynthesizeState(BaseModel) :
    facts : List[str] = Field(...,description="The collection consisting of actual facts and relevant information enriched for effictive writing only without any noise in data.")

class CrtiqueState(BaseModel) :
    is_approved : bool = Field(...,description="Analysis result of the written report if it is remarkably good enough or not.")
    improvements : List[str] = Field(... , description="Strictly mentioning of points of improvemets needed in the current report.")
    fallback_agent: str = Field(...,description="It return the name of the agent to fallback to for improvement.")

class ResearchOutput(TypedDict) :
    objective: str
    source : List[str] 
    content : List[str]

class FactOutput(TypedDict):
    objective: str
    facts: List[str]

class State(TypedDict) :
    user_query : str
    objectives : List[str]
    current_objective : str
    research_output : Annotated[List[ResearchOutput] , operator.add]
    current_research : List[ResearchOutput]
    facts : Annotated[List[FactOutput] , operator.add]
    written_report : str
    is_approved : bool
    improvements : List[str]
    fallback_agent : str
    retry : int 

class ResearchSubState(TypedDict):
    current_objective: str

class SynthesizeSubState(TypedDict):
    individual_research: ResearchOutput

def planner_agent_node(state : State) :
    query = state.get("user_query" , "")
    improvements = state.get("improvements" , ["No improvements needed right now."])

    improvement_context = '\n'.join(improvements)

    prompt = f"""
    <role>
    You are a planner agent in a multi-agent research assistant. Your job is to break the user query into a small sequence of research objectives.
    </role>

    <instructions>
    - Return exactly 3 to 5 objectives.
    - Output only a list of strings.
    - Order the objectives from foundational context to deeper analysis.
    - Each objective must be short, specific, and directly useful for web research and report writing.
    - Make objectives non-overlapping and sequential.
    - If the query is broad, split it into: scope/context, key concepts, evidence/data, comparison/analysis, conclusion implications.
    </instructions>

    <constraints>
    - Do not generate more than 5 objectives.
    - Do not make objectives vague, repetitive, or overly long.
    - Do not shuffle objectives randomly.
    - Do not add explanations or extra text.
    </constraints>
    
    <improvements>
        {improvement_context}
    </improvements>"""

    planner_agent = create_agent(model=llm , system_prompt=prompt , response_format=PlannerState)

    result  = planner_agent.invoke({"messages": [("user", query)]})

    objectives = result.get("structured_response").objectives

    return {
        "objectives" : objectives,
        "research_output": [],
        "facts": []
    }

def research_agent_node(state : ResearchSubState) :
    objective = state.get("current_objective" , "")
    
    prompt = f"""
<role>
You are an enterprise research agent in a multi-agent research assistant.
Your responsibility is to research a single objective and return verified findings.
</role>

<tools>
web_search_tool:
Use this tool to retrieve recent and reliable information.
</tools>

<instructions>
- Research only the assigned objective.
- Collect information from multiple reliable sources.
- Remove duplicate or low-value information.
- Produce concise explanations suitable for downstream report generation.
- Derive insights only from evidence gathered during research.
</instructions>

<output>
{{
    "source": ["url1", "url2" , ...],
    "content": ["researched content 1", "researched content 2" , ...]
}}
</output>

<constraints>
- Do not fabricate information.
- Do not include unsupported claims.
- Do not include irrelevant information.
- Keep findings concise and evidence-driven.
- Cite every finding.
</constraints>
"""

    research_agent = create_agent(model=llm , tools=[web_search_tool] , response_format=ResearchState , system_prompt=prompt)

    response = research_agent.invoke({'messages' : [('user' , objective)]})

    result = response.get("structured_response")

    return {
        'research_output' : [{
            'objective' : objective,
            'source' : result.source,
            'content' : result.content
        }]
    }

def parallel_objective_node(state : State) :
    return [Send("research_agent_node", {"current_objective": obj}) for obj in state.get("objectives", [])]

def synthesizer_agent_node(state : SynthesizeSubState) :
    research_chunk = state.get("individual_research")
    objective = research_chunk.get("objective")
    
    prompt = f"""
<role>
You are a professional synthesizer agent in a enterprise multi-agent research assistant whose job is to read every source and refined content and generate extract facts with citations for effective report writing.
</role>

<input>
Source : Source 1,
Content : Researched Content from source 1


Source : Source 2,
Content : Researched Content from source 2
</input>

<output>
[fact 1 , fact2 ... ]
</output>

<instructions>
- generate clean, concise and knowledge enriched facts extracted from the given input for effective report writing.
- provide citations with very extracted and generated facts.
- the output should be clean and understandable by a large language model 
- every fact should be new, spontaneous and different in meaning
- the number of facts generated should be ideal
</instructions>

<constraints>
- Do not generate duplicated facts
- Do not generate noise, unrelated or vague facts
- The length of the generated should not be overly long
- Do not produce large number of facts 
</constraints>
"""

    query_result = []

    sources = research_chunk.get("source" , [])
    contents = research_chunk.get("content" , [])

    for src , cnt in zip(sources , contents) :
        res = f"Source : {src}\nContent : {cnt}"
        query_result.append(res)

    query = "\n\n".join(query_result)

    synthesizer_agent = create_agent(model=llm , response_format=SynthesizeState , system_prompt=prompt)

    response = synthesizer_agent.invoke({"messages" : [("user" , query)]})

    result = response.get("structured_response").facts

    return {
        "facts" : [{
            "objective": objective,
            "facts": result
        }]
    }

def parallel_research_node(state : State) :
    return [Send("synthesizer_agent_node", {"individual_research": res}) for res in state.get("research_output", [])]

def writer_agent_node(state : State) :
    objectives = state.get("objectives" , [])
    raw_facts = state.get("facts" , [])
    improvements = state.get("improvements" , ["No improvements needed right now."])

    improvement_context = '\n'.join(improvements)

    fact_map = {item["objective"]: item["facts"] for item in raw_facts}

    context_build = []

    for obj in objectives :
        obj_facts = fact_map.get(obj, ["No facts found."])
        fct_str = "\n".join(f"- {f}" for f in obj_facts)
        context_build.append(f"Objective : {obj}\nFacts : \n{fct_str}")

    context = "\n\n".join(context_build)

    prompt = f"""
<role>
You are professional report writer agent present in a enterprise multi-agent research assistant and your job is to create a professional and efficient report for the given context.
</role>

<input>
Objective : objective 1
Facts : facts generated for objective 1
...

Objectives are given in a foundational context to deeper analysis order.
</input>

<report_structure>

# Title

## Executive Summary

## Introduction

## Objective 1

### Explanation

### Key Findings

### Example

## Objective 2

...

## Key Takeaways

## Conclusion

</report_structure>

<instructions>
- Use only the provided facts.
- Preserve citations.
- Follow the objective order exactly.
- Explain concepts clearly.
- Avoid repetition across sections.
- Do not introduce unsupported information.
</instructions>

<constraints>
- The report should not be a random paragraph of words should follow a strict and structured format.
- The report should not be overly long or repetitive.
- The report should follow a order from foundational context to deeper analysis.
</constraints>

<improvements>
        {improvement_context}
</improvements>
"""

    writer_agent = create_agent(model=llm , system_prompt=prompt)

    result = writer_agent.invoke({"messages" : [("user" , context)]})

    return {
        'written_report' : result["messages"][-1].content
    }

def critique_agent_node(state : State) :
    objectives = state.get("objectives" , [])
    report = state.get("written_report" , "")
    current_retry = state.get("retry", 0)

    objectives_context = " ".join(objectives)

    query = f"Objectives : {objectives_context}\nReport :\n{report}\n"

    prompt = """
<role>
You are a quality assurance and critique agent in an enterprise multi-agent research assistant.

Your responsibility is to evaluate the final report against the original objectives and determine whether the report is sufficiently complete, accurate, and useful.

You are NOT a perfectionist reviewer.

Your goal is to identify only significant issues that materially reduce report quality. Minor writing imperfections, small stylistic issues, or opportunities for improvement should NOT cause rejection.
</role>

<agents>

Planner:
Responsible for breaking the user query into logical and sequential research objectives.

Researcher:
Responsible for conducting research for a given objective and gathering evidence.

Synthesizer:
Responsible for extracting factual findings and preserving citations from research results.

Writer:
Responsible for transforming objectives and facts into a structured professional report.

</agents>

<input>

Objectives:
[List of objectives generated by the planner]

Report:
[Final report generated by the writer]

</input>

<evaluation_criteria>

Approve the report if:

- All major objectives are addressed.
- The report follows a logical structure.
- The report is understandable and useful.
- The report contains sufficient information to answer the original research goals.
- Any issues found are minor and do not significantly impact quality.

Reject the report only if one or more severe problems exist:

- One or more major objectives are completely missing.
- The report contains major contradictions.
- Large sections are irrelevant to the objectives.
- The report is poorly structured to the point of reducing usability.
- Critical factual content appears missing.
- The report is substantially incomplete.
- The report appears corrupted, nonsensical, or extremely low quality.

</evaluation_criteria>

<fallback_selection>

If rejection is required, identify the most likely source of failure.

Return:

planner
    - objectives are missing, poorly ordered, too broad, too vague, or fail to cover the user request

researcher
    - major information required for objectives is missing

synthesizer
    - important facts were lost, duplicated excessively, merged incorrectly, or citations were not preserved

writer
    - report structure, clarity, organization, or presentation is the primary problem

If uncertain, prefer writer as the fallback agent.

</fallback_selection>

<instructions>

- Be lenient.
- Prefer approval whenever the report reasonably satisfies its objectives.
- Do not reject for minor grammar issues.
- Do not reject for stylistic preferences.
- Do not reject for small improvements that could make the report better.
- Reject only when meaningful deficiencies exist.
- Keep feedback concise and actionable.
- Focus on major quality concerns only.

</instructions>

<output>

{
    "is_approved": true | false,
    "improvements": [
        "improvement 1",
        "improvement 2"
    ],
    "fallback_agent": "planner" | "researcher" | "synthesizer" | "writer" | "END"
}

</output>

<constraints>

- If the report is approved, fallback_agent must be END.
- If the report is approved, improvements_needed should contain only optional improvements.
- Do not invent missing requirements that are not present in the objectives.
- Do not request rewrites for minor issues.
- Default to approval unless serious problems are detected.

</constraints>"""

    critique_agent = create_agent(model=llm , system_prompt=prompt , response_format=CrtiqueState)

    response = critique_agent.invoke({"messages" : [("user" , query)]})

    result = response.get("structured_response")

    return {
        'is_approved' :  result.is_approved, 
        'improvements' : result.improvements,
        'fallback_agent' : result.fallback_agent,
        'retry' : current_retry + 1
    }

def route_after_critique(state: State):
    if state.get("is_approved") or state.get("retry", 0) >= 2:
        return END
        
    agent_target = state.get("fallback_agent", "writer").lower()
    
    if agent_target == "planner":
        return "planner_agent_node"
        
    elif agent_target == "researcher":
        return [Send("research_agent_node", {"current_objective": obj}) 
                for obj in state.get("objectives", [])]
                
    elif agent_target == "synthesizer":
        return [Send("synthesizer_agent_node", {"current_research": res}) 
                for res in state.get("research_output", [])]
                
    else:
        return "writer_agent_node"
    

workflow = StateGraph(State)
    
workflow.add_node("planner_agent_node", planner_agent_node)
workflow.add_node("research_agent_node", research_agent_node)
workflow.add_node("synthesizer_agent_node", synthesizer_agent_node)
workflow.add_node("writer_agent_node", writer_agent_node)
workflow.add_node("critique_agent_node", critique_agent_node)

workflow.add_edge(START, "planner_agent_node")
workflow.add_conditional_edges("planner_agent_node", parallel_objective_node)
workflow.add_conditional_edges("research_agent_node", parallel_research_node)
workflow.add_edge("synthesizer_agent_node", "writer_agent_node")
workflow.add_edge("writer_agent_node", "critique_agent_node")
workflow.add_conditional_edges("critique_agent_node", route_after_critique)

agent = workflow.compile()