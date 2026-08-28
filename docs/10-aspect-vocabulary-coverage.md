# 10. Aspect vocabulary coverage

Which harness cells have an aspect vocabulary from the collaborators' package, which do not,
and why. Generated from `data/aspects/` and the `aspect_prompt` heldout run; every number here
is computed rather than transcribed.

## What a vocabulary is

An aspect vocabulary is a per-cluster list of the evaluative dimensions through which a topic is
discussed, each with a name, a description, and a count of how many posts on each platform
invoked it. For the Kubernetes cluster the entries include deployment complexity, system
reliability, and performance efficiency. It is not a token vocabulary and has nothing to do with
what the language model can express. A cell without one simply has no extracted list of aspects,
so `aspect_prompt` has nothing to condition on and falls back to zero-shot rewriting.

## Headline

- 19 of 30 harness cells have a vocabulary.
- 19 of 36 shipped clusters match no harness cell and are unused.
- On the heldout run, 268 of 504 generated posts were aspect-conditioned; 200 fell back for want of a vocabulary and 36 because no aspect cleared the threshold.

The gap is not simply missing work. Clusters were extracted under labels that do not match the
harness `final_topic` labels, so vocabularies that exist are not being found.

## Cells with no vocabulary

| Cell | Domain | Topic | LinkedIn | Reddit | Heldout posts affected |
|---|---|---|---|---|---|
| `fullstack_engineer::career_development` | Career | career development | 147 | 9 | 104 |
| `edtech_engineering::online_learning_courses` | EdTech | online learning courses | 5 | 56 | 4 |
| `fullstack_engineer::software_development_tools` | Software Engineering | software development tools | 32 | 17 | 20 |
| `backend_engineer::elixir_programming_language` | Software Engineering | elixir programming language | 17 | 20 | 20 |
| `backend_engineer::ruby_on_rails_releases` | Open Source | ruby on rails releases | 6 | 31 | 4 |
| `backend_engineer::open_source_software` | Open Source | open source software | 16 | 14 | 0 |
| `fullstack_engineer::api_development` | Software Engineering | api development | 8 | 6 | 8 |
| `product_leadership::artificial_intelligence` | AI | artificial intelligence | 6 | 7 | 0 |
| `cto::engineering_management_challenges` | Leadership & Management | engineering management challenges | 5 | 7 | 20 |
| `edtech_engineering::education_technology_jobs` | EdTech | education technology jobs | 6 | 6 | 0 |
| `fullstack_engineer::open_source_software` | Open Source | open source software | 5 | 7 | 20 |

`fullstack_engineer::career_development` alone accounts for 104 of the 200 fallbacks and is the largest cell in the corpus.

## Shipped clusters that match no cell

| Cluster | Aspects | LinkedIn | Reddit |
|---|---|---|---|
| artificial intelligence adoption | 8 | 119 | 18 |
| career decision making | 8 | 109 | 32 |
| cloud and edge compute | 8 | 9 | 4 |
| elixir development | 8 | 18 | 20 |
| file storage | 8 | 6 | 9 |
| gpt 5 models | 10 | 22 | 15 |
| graph technology | 8 | 6 | 36 |
| kubernetes | 8 | 29 | 15 |
| modal state management | 8 | 3 | 5 |
| open source maintenance | 8 | 42 | 28 |
| oracle to postgresql migration | 8 | 22 | 4 |
| postgresql ecosystem | 8 | 110 | 9 |
| pull request review | 8 | 6 | 3 |
| sentry platform | 8 | 39 | 3 |
| seo audit tools | 8 | 6 | 25 |
| software development practices | 8 | 9 | 15 |
| sql learning and usage | 8 | 14 | 14 |
| startup entrepreneurship | 8 | 27 | 13 |
| tech job hiring | 8 | 21 | 3 |

## Probable label mismatches

Matching each uncovered topic against the aspect names and descriptions of each unused cluster,
rather than against the cluster label alone, gives these candidates. They are suggestions for
the collaborators to confirm, not conclusions; the pairing should be checked against the posts
before any vocabulary is remapped.

| Uncovered harness topic | Probable shipped cluster | Cell posts |
|---|---|---|
| artificial intelligence | artificial intelligence adoption | 13 |
| career development | career decision making | 156 |
| elixir programming language | elixir development | 37 |
| open source software | open source maintenance | 42 |
| software development tools | software development practices | 49 |

Confirming these would move four cells from fallback to conditioned, including the two largest
uncovered ones. The remaining uncovered topics have no plausible counterpart in the shipped set
and need extraction: `ruby on rails releases`, `engineering management challenges`,
`education technology jobs`, `online learning courses`, and `api development`.

## Cells with a vocabulary

| Cell | Topic | Aspects | LinkedIn | Reddit |
|---|---|---|---|---|
| `backend_engineer::agentic_coding` | agentic coding | 8 | 15 | 5 |
| `backend_engineer::ai_agents` | ai agents | 8 | 8 | 5 |
| `backend_engineer::ai_software_development` | ai software development | 8 | 23 | 5 |
| `backend_engineer::code_editors` | code editors | 9 | 9 | 5 |
| `backend_engineer::data_structures_and_algorithms` | data structures and algorithms | 8 | 12 | 5 |
| `backend_engineer::docker_container_technology` | docker container technology | 10 | 6 | 8 |
| `backend_engineer::python_ecosystem` | python ecosystem | 8 | 10 | 15 |
| `backend_engineer::react_state_management` | react state management | 8 | 6 | 5 |
| `backend_engineer::ruby_on_rails_ecosystem` | ruby on rails ecosystem | 8 | 21 | 96 |
| `backend_engineer::rust_programming_language` | rust programming language | 8 | 7 | 18 |
| `backend_engineer::software_developer_jobs` | software developer jobs | 8 | 6 | 10 |
| `backend_engineer::software_testing` | software testing | 9 | 10 | 9 |
| `backend_engineer::sql_query_optimization` | sql query optimization | 8 | 14 | 11 |
| `backend_engineer::vector_search` | vector search | 8 | 17 | 6 |
| `edtech_engineering::ai_agents` | ai agents | 8 | 5 | 5 |
| `edtech_engineering::ai_in_education` | ai in education | 9 | 5 | 65 |
| `edtech_engineering::student_emotional_experience` | student emotional experience | 8 | 5 | 30 |
| `instructional_designer::accessible_learning` | accessible learning | 8 | 5 | 5 |
| `instructional_designer::student_emotional_experience` | student emotional experience | 8 | 7 | 7 |

## What to ask for

1. Confirm or reject the five probable label mismatches above, and supply the mapping used to
   name clusters so the join can be made on a stable key rather than on a topic string.
2. Extract vocabularies for the five genuinely uncovered topics, prioritising
   `online learning courses` and `ruby on rails releases`, which carry 56 and 31 Reddit posts.
3. Supply the extraction for `fullstack_engineer::career_development` under whichever label it
   was clustered as. At 147 LinkedIn posts it is the largest cell in the corpus and currently
   contributes nothing to any aspect-conditioned result.

Every figure above is provisional in one further respect: the extraction ran with platform
labels visible to the extracting model, and a blind rerun is pending.

