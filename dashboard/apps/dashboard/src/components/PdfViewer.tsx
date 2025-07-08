import React from 'react';

interface PdfViewerProps {
  framework: any; // Replace 'any' with the appropriate type for your framework
}

const PdfViewer: React.FC<PdfViewerProps> = ({ framework }) => {
  return (
    <div className="pdf-viewer">
      <div className="aspect-w-4 aspect-h-5 bg-gray-100 rounded-lg overflow-hidden">
        <div className="flex items-center justify-center h-full">
          <p className="text-gray-500">PDF Viewer for {framework?.name || 'selected framework'}</p>
        </div>
      </div>
    </div>
  );
};

export default PdfViewer;
