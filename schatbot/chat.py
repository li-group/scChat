#working
import openai

openai.api_key = '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'

def ask(question, session_prompt="The following is a conversation with an AI assistant."):
    try:
        print ("NOW AT ASK FUNCTION : ", question)
        gpt_model = 'gpt-3.5-turbo'
        response = openai.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": session_prompt},
                {"role": "user", "content": question}
            ]
        )

        # print happense until here for the input question
        message_content = response.choices[0].message.content
        print ("NOW AT MESSAGE CONTENT : ", message_content)
        return message_content
    except Exception as e:
      return "I am unable to answer that question right now."

# # def ask(question, session_prompt="The following is a conversation with an AI assistant."):
# #     try:
# #         print("NOW AT ASK FUNCTION :", question)
# #         question = "bullshot what is it>???"
# #         gpt_model = 'gpt-3.5-turbo'
# #         response = openai.chat.completions.create(
# #             model=gpt_model,
# #             messages=[
# #                 {"role": "system", "content": session_prompt},
# #                 {"role": "user", "content": question}
# #             ],
# #             stream = True
# #         )

# #         for choice in response.choices:
# #             yield choice.message.content
# #     except Exception as e:
# #         yield "I am unable to answer that question right now."


