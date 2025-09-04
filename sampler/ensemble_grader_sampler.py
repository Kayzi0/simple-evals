import json
from ..types_eval import MessageList, SamplerBase, SamplerResponse


class EnsembleGraderSampler(SamplerBase):
    """
    Ensemble grader that combines multiple graders using majority vote.
    """

    def __init__(self, graders: list[SamplerBase]):
        self.graders = graders

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        responses = []
        for grader in self.graders:
            resp = grader(message_list)
            try:
                parsed = json.loads(resp.response_text)
            except Exception:
                parsed = {"criteria_met": False, "explanation": "Parse error"}
            responses.append(parsed)

        # Majority vote
        votes = [r.get("criteria_met", False) for r in responses]
        majority = sum(votes) > len(votes) / 2

        # Gather explanations
        explanations = [r.get("explanation", "") for r in responses]
        explanation = " | ".join(explanations)

        result_json = {
            "criteria_met": majority,
            "explanation": explanation,
        }

        return SamplerResponse(
            response_text=json.dumps(result_json),
            response_metadata={"votes": votes, "raw_responses": responses},
            actual_queried_message_list=message_list,
        )
