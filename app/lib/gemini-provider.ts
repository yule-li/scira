import { GoogleGenerativeAI } from '@google/generative-ai';
import { customProvider, wrapLanguageModel, extractReasoningMiddleware, LanguageModelV1, type LanguageModelV1CallOptions, type LanguageModelV1Response } from 'ai';

export const gemini = (model: string) => {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
  const modelInstance = genAI.getGenerativeModel({ 
    model,
    generationConfig: {
      temperature: 0.7,
      topP: 0.8,
      topK: 40,
      maxOutputTokens: 2048,
    }
  });

  const wrappedModel: LanguageModelV1 = {
    provider: 'google',
    modelId: model,
    async doGenerate(options: LanguageModelV1CallOptions): Promise<LanguageModelV1Response> {
      const result = await modelInstance.generateContent(options.prompt.toString());
      return {
        text: result.response.text(),
        reasoning: undefined,
        toolCalls: undefined,
        logprobs: undefined
      };
    },
    async doGenerateStream(options: LanguageModelV1CallOptions) {
      const result = await modelInstance.generateContentStream(options.prompt.toString());
      return result.stream;
    }
  };

  return wrapLanguageModel({
    model: wrappedModel,
    middleware: extractReasoningMiddleware({ tagName: 'think' }),
  });
};
