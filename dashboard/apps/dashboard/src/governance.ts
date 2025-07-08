import type { GovernanceLayer } from './types';

export const DOCUMENT_FRAMEWORKS: GovernanceLayer[] = [
  {
    id: 'constitution',
    name: 'Constitutional Principles',
    description: 'Fundamental governing principles for ethical AI behavior',
    rules: [
      {
        id: 'PRINCIPLE_1',
        title: 'Beneficence',
        description: 'Act in the best interest of humanity',
        strength: 'strong'
      },
      {
        id: 'PRINCIPLE_2',
        title: 'Non-maleficence',
        description: 'Avoid causing harm through action or inaction',
        strength: 'strong'
      },
      {
        id: 'PRINCIPLE_3',
        title: 'Autonomy',
        description: 'Respect human decision-making and consent',
        strength: 'moderate'
      },
      {
        id: 'PRINCIPLE_4',
        title: 'Justice',
        description: 'Ensure fair distribution of benefits and burdens',
        strength: 'moderate'
      }
    ]
  }
];

export const injectionPatterns = [
  /\b(?:password|secret|key|token)\s*[=:]\s*[\'\"][^\'\"]+[\'\"]/gi,
  /\b(?:\$\{?[A-Z_]+\}?)\b/gi
];
