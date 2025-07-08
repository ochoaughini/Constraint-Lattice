import React from 'react';
import { DocumentFrameworkId } from '../../types';

interface FrameworkPdfViewerProps {
  framework: DocumentFrameworkId;
}

export default function FrameworkPdfViewer({ framework }: FrameworkPdfViewerProps) {
  return (
    <div className="p-4 bg-gray-100 rounded-lg">
      PDF Viewer Placeholder for {framework}
    </div>
  );
}
