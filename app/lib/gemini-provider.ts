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
    id: model,
    name: 'Gemini Flash 2.0',
    vendor: 'google',
    version: '1',
    contextWindow: 32000,
    async complete(prompt: string) {
      const result = await modelInstance.generateContent(prompt);
      return result.response.text();
    },
    async completeStream(prompt: string) {
      const result = await modelInstance.generateContentStream(prompt);
      return result.stream;
    }
  };

  return wrapLanguageModel({
    model: wrappedModel,
    middleware: extractReasoningMiddleware({ tagName: 'think' }),
  });
};
