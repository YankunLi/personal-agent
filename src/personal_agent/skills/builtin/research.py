"""Built-in research skill example."""

from personal_agent.skills.base import Skill

RESEARCH_SKILL = Skill(
    name="research",
    description="Deep research capability: search, analyze, and synthesize information from multiple sources",
    prompt="""## Research Skill

You have deep research capabilities. When researching a topic:

1. **Search broadly**: Use web_search to find information from multiple sources
2. **Verify facts**: Cross-reference information across sources
3. **Synthesize**: Combine findings into a coherent analysis
4. **Cite sources**: Reference where information came from
5. **Identify gaps**: Note what information is missing or uncertain

Be thorough and systematic in your research approach.""",
    tools=[],  # Uses the global tool registry's web_search
    dependencies=[],
)

__all__ = ["RESEARCH_SKILL"]