import { useRef } from 'react';

import { useAppStore } from '../../store/appStore';


export default function PdfUpload() {
  const inputRef = useRef(null);
  const isLoading = useAppStore((state) => state.loading.ingestion);
  const ingestPdfs = useAppStore((state) => state.ingestPdfs);

  const handleFilesSelected = async (event) => {
    const selectedFiles = Array.from(event.target.files || []);
    await ingestPdfs(selectedFiles);
    event.target.value = '';
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    if (isLoading) {
      return;
    }
    await ingestPdfs(Array.from(event.dataTransfer.files || []));
  };

  return (
    <label
      className={`upload-zone${isLoading ? ' upload-zone-disabled' : ''}`}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        aria-label="Upload PDF source documents"
        disabled={isLoading}
        onChange={handleFilesSelected}
      />
      <strong>{isLoading ? 'Indexing sources...' : 'Drop or select PDFs'}</strong>
      <span>PDF files are indexed into the active workspace.</span>
    </label>
  );
}
