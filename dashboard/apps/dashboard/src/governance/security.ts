export const injectionPatterns: RegExp[] = [
  /(?:\b|\W)(?:malicious|exploit|inject|bypass|hack|attack)(?:\b|\W)/gi,
  /<script[^>]*>([\s\S]*?)<\/script>/gi,
  /\b(?:document\.|window\.|eval\(|alert\(|fetch\(|XMLHttpRequest)\b/gi,
];
