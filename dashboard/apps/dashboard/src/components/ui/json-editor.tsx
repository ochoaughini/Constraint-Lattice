
import React from 'react';

interface JsonEditorProps {
    value: string;
    onChange: (value: string) => void;
    disabled?: boolean;
}

export const JsonEditor: React.FC<JsonEditorProps> = ({ value, onChange, disabled }) => {
    const lines = value.split('\n').length;
    const lineNumbers = Array.from({ length: lines }, (_, i) => i + 1).join('\n');

    return (
        <div className="flex bg-background border border-white/10 rounded-md overflow-hidden font-mono text-sm">
            <div className="p-3 bg-ui-primary text-right text-text-tertiary select-none">
                <pre>{lineNumbers}</pre>
            </div>
            <textarea
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full p-3 bg-transparent focus:outline-none resize-none"
                rows={Math.max(5, lines)}
                disabled={disabled}
                spellCheck="false"
            />
        </div>
    );
};
