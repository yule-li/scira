import { GoogleGenerativeAI } from '@google/generative-ai';
import { customProvider, wrapLanguageModel, extractReasoningMiddleware, LanguageModelV1, type LanguageModelV1CallOptions } from 'ai';

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
    async doGenerate(options: LanguageModelV1CallOptions) {
      const result = await modelInstance.generateContent(options.prompt.toString());
      const text = result.response.text();
      const response: {
        text?: string;
        reasoning?: string;
        toolCalls?: any[];
        logprobs?: any;
        raw?: any;
        tokens?: any;
        functionCallResult?: any;
        selectedFunctionCall?: any;
        choices?: any;
      } = {
        text: text ?? undefined
      };
      return response;
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
