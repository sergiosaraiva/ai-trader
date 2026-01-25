import { ChevronDown, ChevronUp } from 'lucide-react';
import PropTypes from 'prop-types';
import { useId, memo } from 'react';
import { useCollapsible } from '../../hooks/useCollapsible';

/**
 * Collapsible card component with consistent styling and expand/collapse functionality
 */
export const CollapsibleCard = memo(function CollapsibleCard({
  title,
  icon,
  children,
  defaultExpanded,
  className,
  headerClassName,
  contentClassName,
  actions,
}) {
  const { isExpanded, toggle } = useCollapsible(defaultExpanded);
  const uniqueId = useId();

  return (
    <div className={`bg-gray-800 rounded-lg ${className || ''}`}>
      {/* Header with title and collapse button */}
      <div
        className={`flex items-center justify-between p-4 cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 ${
          headerClassName || ''
        }`}
        onClick={toggle}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggle();
          }
        }}
        aria-expanded={isExpanded}
        aria-controls={`collapsible-content-${uniqueId}`}
      >
        <div className="flex items-center gap-2">
          {icon && <span className="text-blue-400">{icon}</span>}
          <h2 className="text-lg font-bold text-white">{title}</h2>
        </div>

        <div className="flex items-center gap-2">
          {actions && <div className="mr-2">{actions}</div>}
          <button
            type="button"
            className="text-gray-400 hover:text-white transition-colors"
            aria-label={isExpanded ? 'Collapse' : 'Expand'}
            onClick={(e) => {
              e.stopPropagation();
              toggle();
            }}
          >
            {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </button>
        </div>
      </div>

      {/* Collapsible content */}
      <div
        id={`collapsible-content-${uniqueId}`}
        className={`collapsible-content ${isExpanded ? 'expanded' : 'collapsed'} ${
          contentClassName || ''
        }`}
        aria-hidden={!isExpanded}
      >
        <div className="px-4 pb-4">{children}</div>
      </div>
    </div>
  );
});

CollapsibleCard.propTypes = {
  title: PropTypes.string.isRequired,
  icon: PropTypes.node,
  children: PropTypes.node.isRequired,
  defaultExpanded: PropTypes.bool,
  className: PropTypes.string,
  headerClassName: PropTypes.string,
  contentClassName: PropTypes.string,
  actions: PropTypes.node,
};

CollapsibleCard.defaultProps = {
  icon: null,
  defaultExpanded: true,
  className: '',
  headerClassName: '',
  contentClassName: '',
  actions: null,
};

export default CollapsibleCard;
