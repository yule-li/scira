import { GoogleGenerativeAI } from '@google/generative-ai';
import { customProvider, wrapLanguageModel, extractReasoningMiddleware, LanguageModelV1 } from 'ai';

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
    modelId: model,
    provider: 'google',
    async generateText(input: string) {
      const result = await modelInstance.generateContent(input);
      return result.response.text();
    },
    async generateTextStream(input: string) {
      const result = await modelInstance.generateContentStream(input);
      return result.stream;
    }
  };

  return wrapLanguageModel({
    model: wrappedModel,
    middleware: extractReasoningMiddleware({ tagName: 'think' }),
  });
};
