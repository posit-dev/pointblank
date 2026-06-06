# draft_validation_plan


Generate an AI-powered validation plan using Pointblank's DraftValidation class.


``` json
{
  "tool": "draft_validation_plan",
  "arguments": {
    "dataframe_id": ...,
    "model": "anthropic:claude-sonnet-4-5",
    "api_key": ...  // optional
  }
}
```


# Parameters


`dataframe_id: string`  
ID of the DataFrame to generate validation plan for. required

`model: string = "anthropic:claude-sonnet-4-5"`  
AI model in format 'provider:model'.

`api_key: any`  
API key for the model provider.
